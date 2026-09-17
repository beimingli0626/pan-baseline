"""Sequence-transformer baseline: the tour as a causal token stream.

A decoder-only stack over one flat sequence, the memory layout of mainstream VLN/VLA
policies (history as tokens, as in HAMT):

    [kf_1 ... kf_K] [query]        causal over the tour; a query reads the whole tour

Tokens are indexed by their rank in the tour with 1-D RoPE, a query at rank K right after the
last keyframe. Geometry enters as content: keyframe t reads its summary, the motion from
keyframe t-1 (unless `motion_input` is off, which removes this frame-to-frame adjacency input) and
where it is (frame_features: position and yaw relative to the first keyframe in world axes, the goal's
offset and the goal seen from the pose); a query reads where it is. The query's top-layer feature goes
to the Planner's autoregressive chunk decoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pan.planner.model import ChunkDecoder, RoPE, Summarizer, scatter_packed
from pan.util.config import arg

from .recurrent import FRAME_FEATURES, MOTION_FEATURES, frame_features, keyframe_motion


@dataclass
class SeqFormerConfig:
    summary_dim: int = arg(384, "width of the per-keyframe summary token", flag="--summary-dim")
    model_dim: int = arg(768, "decoder width", flag="--model-dim")
    layers: int = arg(10, "decoder blocks")
    num_heads: int = arg(12, "attention heads")
    ffn_ratio: int = arg(4, "MLP width over model width")
    chunk: int = arg(8, "actions decoded per query", flag="--chunk")
    act_emb_dim: int = arg(64, "action embedding width of the chunk decoder", flag="--act-emb-dim")
    pose_scale: float = arg(10.0, "metres per unit of the motion, position and goal inputs")
    motion_input: bool = arg(True, "feed each keyframe its motion from the previous keyframe, a frame-to-frame "
                                   "adjacency input; off, a keyframe reads only its summary and frame_features")
    max_steps: int = arg(2048, "unused: sized the keyframe index table that 1-D RoPE replaced")


class Block(nn.Module):
    """Pre-LN decoder block with two entry points over one set of weights: `encode` streams
    the tour causally and returns its rotated keys and values; `decode` lets queries attend to
    those, so an item's queries share one pass over the tour."""

    def __init__(self, d: int, heads: int, ffn_ratio: int, rope: RoPE):
        super().__init__()
        assert d % heads == 0, f"model_dim {d} must divide into {heads} heads"
        self.num_heads, self.head_dim, self.rope = heads, d // heads, rope
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn_ratio * d), nn.GELU(),
                                 nn.Linear(ffn_ratio * d, d))

    def _qkv(self, x: torch.Tensor):
        """x (B, N, d) -> q, k, v (B, H, N, dh)."""
        B, N, _ = x.shape
        return self.qkv(self.ln1(x)).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

    def _update(self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(y.transpose(1, 2).flatten(-2))
        return x + self.mlp(self.ln2(x))

    def encode(self, x: torch.Tensor, rank: torch.Tensor, valid: torch.Tensor):
        """x (B, K, d) at ranks (B, K), valid (B, K) -> (y (B, K, d), keys, values (B, H, K, dh))."""
        B, K, _ = x.shape
        q, k, v = self._qkv(x)
        q, k = self.rope(q, rank), self.rope(k, rank)
        causal = torch.ones(K, K, dtype=torch.bool, device=x.device).tril()
        m = causal.view(1, 1, K, K) & valid.view(B, 1, 1, K)
        m = m | ~m.any(-1, keepdim=True)             # unmask padded rows (no NaN); never read
        return self._update(x, q, k, v, m), k, v

    def decode(self, x: torch.Tensor, rank: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
               valid: torch.Tensor) -> torch.Tensor:
        """Queries x (B, T, d) at ranks (B, T) attend to the tour's keys and values (B, H, K, dh)."""
        q = self.rope(self._qkv(x)[0], rank)
        return self._update(x, q, keys, values, valid.view(valid.shape[0], 1, 1, -1))


class SeqFormer(nn.Module):
    def __init__(self, cfg: SeqFormerConfig = SeqFormerConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.model_dim
        self.summarizer = Summarizer(cfg.summary_dim)
        self.summary_norm = nn.LayerNorm(cfg.summary_dim)
        motion = MOTION_FEATURES if cfg.motion_input else 0
        self.keyframe_proj = nn.Linear(cfg.summary_dim + motion + FRAME_FEATURES, d)
        self.query_proj = nn.Linear(FRAME_FEATURES, d)
        self.query_token = nn.Parameter(torch.randn(d) * 0.02)
        rope = RoPE(d // cfg.num_heads, spatial=False)
        self.blocks = nn.ModuleList(Block(d, cfg.num_heads, cfg.ffn_ratio, rope) for _ in range(cfg.layers))
        self.ln_f = nn.LayerNorm(d)
        self.decoder = ChunkDecoder(d, cfg.act_emb_dim, cfg.chunk)

    def summarize(self, kf_tokens: torch.Tensor, kf_valid: torch.Tensor) -> torch.Tensor:
        """PACKED patch tokens (n_valid, P, Dt) -> (B, K, Ds) normalized summaries."""
        return scatter_packed(self.summary_norm(self.summarizer(kf_tokens)), kf_valid)

    def encode_map(self, entries: torch.Tensor, kf_pose: torch.Tensor, kf_valid: torch.Tensor,
                   goal_position: torch.Tensor):
        """Stream the tour through the stack once, keyframe t at rank t -> (top-layer outputs
        (B, K, d), per-block keys and values (B, H, K, dh), valid (B, K), the anchor: pose of the
        first keyframe (B, 3))."""
        B, K, _ = entries.shape
        s, anchor = self.cfg.pose_scale, kf_pose[:, 0]
        parts = [entries.float(), frame_features(kf_pose, anchor[:, None], goal_position[:, None], s)]
        if self.cfg.motion_input:
            parts.insert(1, keyframe_motion(kf_pose, s))
        x = self.keyframe_proj(torch.cat(parts, -1))
        rank = torch.arange(K, device=x.device, dtype=torch.float32).expand(B, K)
        keys, values = [], []
        for block in self.blocks:
            x, k, v = block.encode(x, rank, kf_valid)
            keys.append(k)
            values.append(v)
        return x, tuple(keys), tuple(values), kf_valid, anchor

    def read(self, memory, query_pose: torch.Tensor, goal_position: torch.Tensor,
             teacher_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Queries at query_pose (B, T, 3), at the rank right after the last keyframe, read the tour
        through every block -> the Planner's chunk decoder -> logits (B, T, chunk, A)."""
        _, keys, values, valid, anchor = memory
        B, T = query_pose.shape[:2]
        x = self.query_token + self.query_proj(
            frame_features(query_pose, anchor[:, None], goal_position[:, None], self.cfg.pose_scale))
        rank = valid.sum(1, keepdim=True).float().expand(B, T)
        for block, k, v in zip(self.blocks, keys, values):
            x = block.decode(x, rank, k, v, valid)
        return self.decoder(self.ln_f(x), teacher_actions)

    def forward(self, kf_tokens, kf_pose, kf_valid, query_pose, goal_position,
                teacher_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        memory = self.encode_map(self.summarize(kf_tokens, kf_valid), kf_pose, kf_valid,
                                 goal_position)
        return self.read(memory, query_pose, goal_position, teacher_actions)
