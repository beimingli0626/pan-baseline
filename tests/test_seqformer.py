"""Sequence-transformer baseline: a causal tour stream indexed by rank, independent queries, the
Planner's chunk decoder, and positions compared in world axes as the Planner does."""
import math

import torch
import torch.nn as nn

from baselines.recurrent import FRAME_FEATURES, MOTION_FEATURES
from baselines.seqformer import SeqFormer, SeqFormerConfig
from pan.util.constants import N_ACTIONS


def _model():
    torch.manual_seed(0)
    m = SeqFormer(SeqFormerConfig(summary_dim=64, model_dim=64, layers=2, num_heads=4, chunk=8,
                                  act_emb_dim=16)).eval()
    nn.init.normal_(m.decoder.out.weight)          # O(1) logits, so the tolerances bite
    return m


def _batch(B=2, K=6, T=3):
    g = torch.Generator().manual_seed(1)
    tok = torch.randn(B, K, 5, 384, generator=g)
    valid = torch.ones(B, K, dtype=torch.bool)
    pose = torch.randn(B, K, 3, generator=g)
    query = torch.randn(B, T, 3, generator=g)
    goal = torch.randn(B, 2, generator=g)
    return tok, pose, valid, query, goal


def _targets(B=2, T=3):
    return torch.randint(0, N_ACTIONS, (B, T, 8), generator=torch.Generator().manual_seed(2))


def test_shapes_and_padding_invariance():
    m, (tok, pose, valid, query, goal) = _model(), _batch()
    tgt = _targets()
    with torch.no_grad():
        out = m(tok[valid], pose, valid, query, goal, teacher_actions=tgt)
        tok_p = torch.cat([tok, torch.randn(2, 2, 5, 384)], 1)
        pose_p = torch.cat([pose, torch.randn(2, 2, 3)], 1)
        valid_p = torch.cat([valid, torch.zeros(2, 2, dtype=torch.bool)], 1)
        padded = m(tok_p[valid_p], pose_p, valid_p, query, goal, teacher_actions=tgt)
    assert out.shape == (2, 3, 8, N_ACTIONS)
    assert torch.allclose(out, padded, atol=1e-5), "padded keyframes changed the answer"


def test_tour_is_causal():
    """Keyframe t must not see t+1."""
    m, (tok, pose, valid, _, goal) = _model(), _batch(B=1, K=6)
    e = m.summarize(tok[valid], valid)
    with torch.no_grad():
        a = m.encode_map(e, pose, valid, goal)[0]
        late = e.clone()
        late[:, 4] += 3.0
        b = m.encode_map(late, pose, valid, goal)[0]
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6), "a later keyframe leaked back"
    assert not torch.allclose(a[:, 4], b[:, 4], atol=1e-6)


def test_query_blocks_are_independent():
    m, (tok, pose, valid, query, goal) = _model(), _batch(T=3)
    tgt = _targets()
    with torch.no_grad():
        a = m(tok[valid], pose, valid, query, goal, teacher_actions=tgt)
        q2, t2 = query.clone(), tgt.clone()
        q2[:, 2] += 5.0
        t2[:, 2] = (t2[:, 2] + 1) % N_ACTIONS
        b = m(tok[valid], pose, valid, q2, goal, teacher_actions=t2)
    assert torch.allclose(a[:, :2], b[:, :2], atol=1e-6), "queries attend to each other"
    assert not torch.allclose(a[:, 2], b[:, 2], atol=1e-6)


def test_decoder_is_causal_and_self_consistent():
    """Slot i conditions on a_1..a_i only, and greedy decoding reproduces teacher forcing on
    its own actions."""
    m, (tok, pose, valid, query, goal) = _model(), _batch()
    tgt = _targets()
    with torch.no_grad():
        a = m(tok[valid], pose, valid, query, goal, teacher_actions=tgt)
        t2 = tgt.clone()
        t2[..., 5] = (t2[..., 5] + 1) % N_ACTIONS
        b = m(tok[valid], pose, valid, query, goal, teacher_actions=t2)
        free = m(tok[valid], pose, valid, query, goal)
        forced = m(tok[valid], pose, valid, query, goal, teacher_actions=free.argmax(-1))
    assert torch.allclose(a[..., :6, :], b[..., :6, :], atol=1e-6), "a later action leaked back"
    assert not torch.allclose(a[..., 6, :], b[..., 6, :], atol=1e-6)
    assert torch.equal(forced.argmax(-1), free.argmax(-1))


def test_world_axes_like_the_planner():
    """Positions are compared in world axes: moving the tour, queries and goal together leaves the
    output unchanged, rotating them together does not."""
    m, (tok, pose, valid, query, goal) = _model(), _batch()
    tgt = _targets()

    def run(th, shift):
        R = torch.tensor([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
        p, q, g = pose.clone(), query.clone(), goal @ R.T + shift
        for x in (p, q):
            x[..., :2] = x[..., :2] @ R.T + shift
            x[..., 2] += th
        with torch.no_grad():
            return m(tok[valid], p, valid, q, g, teacher_actions=tgt)

    ref = run(0.0, torch.zeros(2))
    assert torch.allclose(ref, run(0.0, torch.tensor([3.0, -2.0])), atol=1e-4)
    assert not torch.allclose(ref, run(0.7, torch.zeros(2)), atol=1e-3)


def test_motion_input_is_the_only_adjacency_input():
    """Moving keyframe 3 changes the input token of keyframe 3 only; with motion_input on, keyframe 4's
    motion from keyframe 3 changes too."""
    tok, pose, valid, _, goal = _batch(B=1, K=6)
    moved = pose.clone()
    moved[:, 3, :2] += 1.0
    for motion_input, expected in ((False, [3]), (True, [3, 4])):
        torch.manual_seed(0)
        m = SeqFormer(SeqFormerConfig(summary_dim=64, model_dim=64, layers=2, num_heads=4, chunk=8,
                                      act_emb_dim=16, motion_input=motion_input)).eval()
        inputs = []
        handle = m.keyframe_proj.register_forward_hook(lambda mod, args, out: inputs.append(args[0].detach()))
        with torch.no_grad():
            e = m.summarize(tok[valid], valid)
            m.encode_map(e, pose, valid, goal)
            m.encode_map(e, moved, valid, goal)
        handle.remove()
        assert inputs[0].shape[-1] == 64 + FRAME_FEATURES + (MOTION_FEATURES if motion_input else 0)
        changed = torch.nonzero((inputs[0] - inputs[1]).abs().amax(-1)[0] > 1e-6).flatten().tolist()
        assert changed == expected, (motion_input, changed)


def test_every_parameter_trains():
    m, (tok, pose, valid, query, goal) = _model(), _batch()
    m.train()
    m(tok[valid], pose, valid, query, goal, teacher_actions=_targets()).sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
    assert not dead, f"no gradient reached {dead}"
