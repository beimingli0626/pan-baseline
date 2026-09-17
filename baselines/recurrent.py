"""Recurrent-memory baselines: LSTM and SRU.

They read the same frozen tour keyframes as the Planner but integrate them in tour order
into a fixed-size state.

    lstm      cuDNN LSTM, forget-gate bias 1
    sru       SRU (Yang et al., IJRR 2025): an LSTM whose candidate is modulated by a learned
              linear transform of the input, g = tanh((W_s x + b_s) * (W_g [x, h] + b_g))
    sru_gate  the SRU cell with the refine-gate forget update

Step t reads [summary_t | motion from keyframe t-1 | where keyframe t is]. "Where" is given in
world axes, as the Planner's RoPE compares poses: the position and yaw relative to the first
keyframe, the goal's offset from the position, and the goal seen from the pose. A query is one
more step from the tour's final (h, c), reading a learned no-observation vector, no motion and
where the query is; a residual MLP on that step's output and the Planner's autoregressive chunk
decoder emit the actions. The recurrences run in fp32. With `motion_input` off no step, the query's
included, reads motion columns: the frame-to-frame adjacency input is gone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from pan.planner.model import GOAL_FEATURES, ChunkDecoder, Summarizer, goal_features, scatter_packed
from pan.util.config import arg

CELLS = ("lstm", "sru", "sru_gate")
State = Tuple[torch.Tensor, torch.Tensor]        # (h, c), each (layers, N, hidden)
MOTION_FEATURES = 4
FRAME_FEATURES = 2 + 2 + 2 + GOAL_FEATURES       # position, yaw, goal offset, goal seen from the pose
READOUT_FFN_RATIO = 4                            # the Planner's ffn_ratio


# ---------------------------------------------------------------- pose features
def relative_motion(pose_a: torch.Tensor, pose_b: torch.Tensor,
                    metres_per_unit: float) -> torch.Tensor:
    """Motion a -> b in a's ego frame: (forward, left) / metres_per_unit, sin dyaw, cos dyaw."""
    rel = pose_b[..., :2] - pose_a[..., :2]
    distance = rel.norm(dim=-1)
    bearing = torch.atan2(-rel[..., 0], -rel[..., 1]) - pose_a[..., 2]
    dyaw = pose_b[..., 2] - pose_a[..., 2]
    return torch.stack([distance * torch.cos(bearing) / metres_per_unit,
                        distance * torch.sin(bearing) / metres_per_unit,
                        torch.sin(dyaw), torch.cos(dyaw)], -1)


def keyframe_motion(kf_pose: torch.Tensor, metres_per_unit: float) -> torch.Tensor:
    """(B, K, 3) -> (B, K, 4): each keyframe's relative_motion from its predecessor; the
    first keyframe reads no motion (0, 0, 0, 1)."""
    first = kf_pose.new_zeros(kf_pose.shape[0], 1, MOTION_FEATURES)
    first[..., 3] = 1.0
    if kf_pose.shape[1] == 1:
        return first
    return torch.cat([first, relative_motion(kf_pose[:, :-1], kf_pose[:, 1:], metres_per_unit)], 1)


def frame_features(pose: torch.Tensor, anchor: torch.Tensor, goal_position: torch.Tensor,
                   metres_per_unit: float) -> torch.Tensor:
    """Where `pose` (..., 3) is, in world axes -> (..., FRAME_FEATURES): (position - anchor
    position) / metres_per_unit, sin and cos of (yaw - anchor yaw), (goal_position - position) /
    metres_per_unit, and goal_features. `anchor` and `goal_position` broadcast against `pose`."""
    dyaw = (pose[..., 2] - anchor[..., 2]).unsqueeze(-1)
    return torch.cat([(pose[..., :2] - anchor[..., :2]) / metres_per_unit, torch.sin(dyaw), torch.cos(dyaw),
                      (goal_position - pose[..., :2]) / metres_per_unit,
                      goal_features(pose, goal_position, metres_per_unit)], -1)


@dataclass
class RecurrentConfig:
    cell: str = "lstm"                  # the run's arch
    summary_dim: int = arg(384, "width of the per-keyframe summary token", flag="--summary-dim")
    hidden: int = arg(2048, "recurrent state width")
    layers: int = arg(2, "recurrent layers")
    head_hidden: int = arg(1024, "unused: sized the MLP head that the readout and chunk decoder replaced",
                           flag="--rnn-head-dim")
    readout_dim: int = arg(768, "width of the readout MLP and of the chunk decoder's GRU")
    readout_blocks: int = arg(3, "residual MLP blocks on the query step's output (15.7M parameters with "
                                 "the input projection at the defaults; the Planner's read side has 28.4M)")
    chunk: int = arg(8, "actions decoded per query", flag="--chunk")
    act_emb_dim: int = arg(64, "action embedding width of the chunk decoder", flag="--act-emb-dim")
    pose_scale: float = arg(10.0, "metres per unit of the motion, position and goal inputs")
    motion_input: bool = arg(True, "feed each keyframe its motion from the previous keyframe, a frame-to-frame "
                                   "adjacency input; off, a step reads only its summary and frame_features")
    compile_step: bool = arg(False, "torch.compile the SRU step (stalls under DDP)", flag="--rnn-compile")

    def __post_init__(self):
        assert self.cell in CELLS, f"cell {self.cell!r} not in {CELLS}"


# ---------------------------------------------------------------- SRU
def _sru_cell(a: torch.Tensor, tx: torch.Tensor, c: torch.Tensor, refine: bool):
    """One SRU-LSTM update from the gate pre-activations a = W_x x + b + W_h h (B, 4H) and
    the transform term tx = W_s x + b_s (B, H) -> (h, c, gates [i, f, o, g], a_g)."""
    ai, af, ao, ag = a.chunk(4, -1)
    i, f, o = torch.sigmoid(ai), torch.sigmoid(af), torch.sigmoid(ao)
    g = torch.tanh(tx * ag)
    if refine:                                      # refine gate (Gu et al. 2020)
        fe = i * (1.0 - (1.0 - f) ** 2) + (1.0 - i) * f ** 2
        c_new = fe * c + (1.0 - fe) * g
    else:
        c_new = f * c + i * g
    return o * torch.tanh(c_new), c_new, torch.cat([i, f, o, g], -1), ag


def _sru_step(a, tx, h, c, m, refine: bool):
    """`_sru_cell` in a sequence: padded steps (m (B, 1) False) keep the state."""
    h_new, c_new, gates, ag = _sru_cell(a, tx, c, refine)
    return torch.where(m, h_new, h), torch.where(m, c_new, c), gates, ag


def _sru_back(dh, dc, gates, ag, tx, c_new, c_prev, m, refine: bool):
    """Backward of `_sru_step` for one step, from the total gradients at (h_t, c_t) ->
    (d a (B, 4H), d tx (B, H), d c_{t-1} (B, H)); the caller forms d h_{t-1}."""
    i, f, o, g = gates.chunk(4, -1)
    tc = torch.tanh(c_new)
    dc_new = dc + dh * o * (1.0 - tc * tc)
    da_o = dh * tc * o * (1.0 - o)
    if refine:
        fe = i * (1.0 - (1.0 - f) ** 2) + (1.0 - i) * f ** 2
        dfe = dc_new * (c_prev - g)
        dg = dc_new * (1.0 - fe)
        di = dfe * ((1.0 - (1.0 - f) ** 2) - f ** 2)
        df = dfe * (2.0 * i * (1.0 - f) + 2.0 * (1.0 - i) * f)
        dc_prev_new = dc_new * fe
    else:
        di, df, dg = dc_new * g, dc_new * c_prev, dc_new * i
        dc_prev_new = dc_new * f
    da_i, da_f = di * i * (1.0 - i), df * f * (1.0 - f)
    du = dg * (1.0 - g * g)
    mf = m.to(dh.dtype)
    da = torch.cat([da_i, da_f, da_o, du * tx], -1) * mf
    return da, du * ag * mf, torch.where(m, dc_prev_new, dc)


class _SRURecurrence(torch.autograd.Function):
    """Whole-sequence SRU recurrence with manual BPTT: the (4H x H) recurrent-weight
    gradient is one matmul over the stacked step gradients rather than an autograd add per
    step (which dominated backward at H=2048)."""

    @staticmethod
    def forward(ctx, xp, tx, Wh, h0, c0, mask, refine, step_fn, back_fn):
        B, k, H4 = xp.shape
        H = H4 // 4
        outs, C = xp.new_empty(B, k, H), xp.new_empty(B, k, H)
        G, AG = xp.new_empty(B, k, H4), xp.new_empty(B, k, H)
        WhT = Wh.t()
        h, c = h0, c0
        for t in range(k):
            a = torch.addmm(xp[:, t], h, WhT)
            h, c, g_t, ag_t = step_fn(a, tx[:, t], h, c, mask[:, t], refine)
            outs[:, t], C[:, t], G[:, t], AG[:, t] = h, c, g_t, ag_t
        ctx.save_for_backward(tx, Wh, h0, c0, mask, outs, C, G, AG)
        ctx.refine, ctx.back_fn = refine, back_fn
        return outs, h, c

    @staticmethod
    def backward(ctx, douts, dhK, dcK):
        tx, Wh, h0, c0, mask, outs, C, G, AG = ctx.saved_tensors
        B, k, H = outs.shape
        dA, dTX = torch.empty(B, k, 4 * H, dtype=tx.dtype, device=tx.device), torch.empty_like(tx)
        dh, dc = dhK, dcK
        for t in range(k - 1, -1, -1):
            dh = dh + douts[:, t]
            c_prev = C[:, t - 1] if t > 0 else c0
            da, dtx, dc = ctx.back_fn(dh, dc, G[:, t], AG[:, t], tx[:, t], C[:, t],
                                      c_prev, mask[:, t], ctx.refine)
            dA[:, t], dTX[:, t] = da, dtx
            dh = torch.where(mask[:, t], torch.mm(da, Wh), dh)
        h_prev = torch.cat([h0.unsqueeze(1), outs[:, :-1]], 1).reshape(-1, H)
        dWh = dA.reshape(-1, 4 * H).t().mm(h_prev)
        return dA, dTX, dWh, dh, dc, None, None, None, None


class SRUCell(nn.Module):
    """LSTM cell with the SRU spatial-transformation gate (Yang et al.)."""

    def __init__(self, d_in: int, hidden: int, refine_gate: bool = False):
        super().__init__()
        self.hidden, self.refine = hidden, refine_gate
        self.linear_all = nn.Linear(d_in + hidden, 4 * hidden)
        nn.init.orthogonal_(self.linear_all.weight)
        with torch.no_grad():                       # forget bias 1 (+noise), as upstream
            self.linear_all.bias[hidden:2 * hidden] = 1.0 + torch.randn(hidden)
        self.transform_gate = nn.Linear(d_in, hidden)
        nn.init.orthogonal_(self.transform_gate.weight)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        h, c, _, _ = _sru_cell(self.linear_all(torch.cat([x, h], -1)), self.transform_gate(x),
                               c, self.refine)
        return h, c


class SRUStack(nn.Module):
    """SRUCells run layer-major, like cuDNN: per layer the input projections of all steps
    are one matmul, and the loop keeps only the recurrent matmul and a fused update."""

    def __init__(self, d_in: int, hidden: int, layers: int, refine_gate: bool,
                 compile_step: bool = False):
        super().__init__()
        self.hidden, self.layers, self.refine = hidden, layers, refine_gate
        self.cells = nn.ModuleList(SRUCell(d_in if i == 0 else hidden, hidden, refine_gate)
                                   for i in range(layers))
        if compile_step:
            self._step_fn = torch.compile(_sru_step, dynamic=False)
            self._back_fn = torch.compile(_sru_back, dynamic=False)
        else:
            self._step_fn, self._back_fn = _sru_step, _sru_back

    def forward(self, x: torch.Tensor, valid: torch.Tensor, state: Optional[State] = None):
        """x (B, K, D) fp32, valid (B, K) -> outputs (B, K, H), (h, c) (L, B, H)."""
        B, K, _ = x.shape
        if state is None:
            zeros = x.new_zeros(self.layers, B, self.hidden)
            state = (zeros, zeros.clone())
        h0, c0 = state
        k_eff = int(valid.sum(1).max().item()) if K else 0
        m = valid[:, :k_eff].unsqueeze(-1)
        inp = x[:, :k_eff]
        hs, cs = [], []
        for layer, cell in enumerate(self.cells):
            d_in = inp.shape[-1]
            W = cell.linear_all.weight
            xp = F.linear(inp, W[:, :d_in], cell.linear_all.bias)      # (B, k, 4H)
            tx = cell.transform_gate(inp)                               # (B, k, H)
            Wh = W[:, d_in:].contiguous()
            if k_eff:
                inp, h, c = _SRURecurrence.apply(xp, tx, Wh, h0[layer], c0[layer], m, self.refine,
                                                 self._step_fn, self._back_fn)
            else:
                inp, h, c = x.new_zeros(B, 0, self.hidden), h0[layer], c0[layer]
            hs.append(h)
            cs.append(c)
        if k_eff < K:
            inp = torch.cat([inp, hs[-1].unsqueeze(1).expand(B, K - k_eff, self.hidden)], 1)
        return inp, (torch.stack(hs), torch.stack(cs))

    def step(self, x: torch.Tensor, state: State) -> torch.Tensor:
        """One step of every layer from the state (h, c), x (N, D) -> the top layer's output (N, H)."""
        h, c = state
        for layer, cell in enumerate(self.cells):
            x, _ = cell(x, h[layer], c[layer])
        return x


class LSTMStack(nn.Module):
    def __init__(self, d_in: int, hidden: int, layers: int):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, layers, batch_first=True)
        with torch.no_grad():                       # forget bias 1, split over ih and hh
            for layer in range(layers):
                getattr(self.lstm, f"bias_ih_l{layer}")[hidden:2 * hidden] = 0.5
                getattr(self.lstm, f"bias_hh_l{layer}")[hidden:2 * hidden] = 0.5

    def forward(self, x: torch.Tensor, valid: torch.Tensor, state: Optional[State] = None):
        lengths = valid.sum(1).clamp_min(1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(packed, state)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return out, (h, c)

    def step(self, x: torch.Tensor, state: State) -> torch.Tensor:
        """One step of every layer from the state (h, c), x (N, D) -> the top layer's output (N, H)."""
        h, c = state
        out, _ = self.lstm(x.unsqueeze(1), (h.contiguous(), c.contiguous()))
        return out[:, 0]


# ---------------------------------------------------------------- the planner
class RecurrentPlanner(nn.Module):
    def __init__(self, cfg: RecurrentConfig = RecurrentConfig()):
        super().__init__()
        self.cfg = cfg
        self.summarizer = Summarizer(cfg.summary_dim)
        self.summary_norm = nn.LayerNorm(cfg.summary_dim)
        d_in = cfg.summary_dim + (MOTION_FEATURES if cfg.motion_input else 0) + FRAME_FEATURES
        if cfg.cell == "lstm":
            self.rnn = LSTMStack(d_in, cfg.hidden, cfg.layers)
        else:
            self.rnn = SRUStack(d_in, cfg.hidden, cfg.layers, refine_gate=cfg.cell == "sru_gate",
                                compile_step=cfg.compile_step)
        self.no_observation = nn.Parameter(torch.zeros(cfg.summary_dim))
        d = cfg.readout_dim
        self.readout_proj = nn.Linear(cfg.hidden, d)
        self.readout = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, READOUT_FFN_RATIO * d), nn.GELU(),
                          nn.Linear(READOUT_FFN_RATIO * d, d))
            for _ in range(cfg.readout_blocks))
        self.read_norm = nn.LayerNorm(d)
        self.decoder = ChunkDecoder(d, cfg.act_emb_dim, cfg.chunk)

    def summarize(self, kf_tokens: torch.Tensor, kf_valid: torch.Tensor) -> torch.Tensor:
        """PACKED patch tokens (n_valid, P, Dt) -> (B, K, Ds) normalized summaries."""
        return scatter_packed(self.summary_norm(self.summarizer(kf_tokens)), kf_valid)

    def encode_map(self, entries: torch.Tensor, kf_pose: torch.Tensor, kf_valid: torch.Tensor,
                   goal_position: torch.Tensor):
        """Run the recurrence over the tour, step t reading [summary | motion from keyframe
        t-1 | frame_features of keyframe t] -> (per-keyframe top-layer outputs (B, K, H), final
        h and c (L, B, H), the anchor: pose of the first keyframe (B, 3))."""
        s, anchor = self.cfg.pose_scale, kf_pose[:, 0]
        parts = [entries.float(), frame_features(kf_pose, anchor[:, None], goal_position[:, None], s)]
        if self.cfg.motion_input:
            parts.insert(1, keyframe_motion(kf_pose, s))
        x = torch.cat(parts, -1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            tokens, (h, c) = self.rnn(x.float(), kf_valid)
        return tokens, h, c, anchor

    def read(self, memory, query_pose: torch.Tensor, goal_position: torch.Tensor,
             teacher_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Each query at query_pose (B, T, 3) is one recurrent step from the tour's final (h, c),
        reading [no-observation vector | no motion | frame_features of the query]; a residual MLP
        on the step's output, then the Planner's chunk decoder -> logits (B, T, chunk, A)."""
        _, h, c, anchor = memory
        (B, T), (L, _, H) = query_pose.shape[:2], h.shape
        where = frame_features(query_pose, anchor[:, None], goal_position[:, None], self.cfg.pose_scale)
        parts = [self.no_observation.expand(B, T, -1), where]
        if self.cfg.motion_input:
            motion = query_pose.new_zeros(B, T, MOTION_FEATURES)
            motion[..., 3] = 1.0
            parts.insert(1, motion)
        x = torch.cat(parts, -1).reshape(B * T, -1)
        state = tuple(t.unsqueeze(2).expand(L, B, T, H).reshape(L, B * T, H) for t in (h, c))
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self.rnn.step(x.float(), state)
        x = self.readout_proj(out).view(B, T, -1)
        for block in self.readout:
            x = x + block(x)
        return self.decoder(self.read_norm(x), teacher_actions)

    def forward(self, kf_tokens, kf_pose, kf_valid, query_pose, goal_position,
                teacher_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        memory = self.encode_map(self.summarize(kf_tokens, kf_valid), kf_pose, kf_valid,
                                 goal_position)
        return self.read(memory, query_pose, goal_position, teacher_actions)
