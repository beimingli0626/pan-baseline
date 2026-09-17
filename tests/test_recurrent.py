"""Recurrent-memory baselines (LSTM / SRU): interface parity with the Planner, padding
invariance, gradient coverage, world-axis pose inputs, manual BPTT."""
import dataclasses
import math

import pytest
import torch
import torch.nn as nn

from baselines.recurrent import FRAME_FEATURES, MOTION_FEATURES, RecurrentConfig, RecurrentPlanner, SRUStack
from pan.planner.model import Planner, PlannerConfig, goal_features
from pan.util.constants import N_ACTIONS

CELLS = ("lstm", "sru", "sru_gate")


def _cfg(cell):
    return RecurrentConfig(cell=cell, summary_dim=32, hidden=24, layers=2, readout_dim=24,
                           readout_blocks=1, act_emb_dim=8, chunk=8)


def _batch(B=2, K=7, T=3, seed=0):
    """tok DENSE (B, K, P, Dt) for row-wise manipulation; the model takes tok[valid]."""
    g = torch.Generator().manual_seed(seed)
    tok = torch.randn(B, K, 5, 384, generator=g)
    pose = torch.cumsum(torch.randn(B, K, 3, generator=g) * 0.3, 1)
    valid = torch.ones(B, K, dtype=torch.bool)
    query = torch.randn(B, T, 3, generator=g)
    goal = torch.randn(B, 2, generator=g)
    return tok, pose, valid, query, goal


@pytest.mark.parametrize("cell", CELLS)
def test_shapes_and_memory(cell):
    m = RecurrentPlanner(_cfg(cell)).eval()
    tok, pose, valid, query, goal = _batch()
    out = m(tok[valid], pose, valid, query, goal)
    memory = m.encode_map(m.summarize(tok[valid], valid), pose, valid, goal)
    assert out.shape == (2, 3, 8, N_ACTIONS) and torch.isfinite(out).all()
    assert memory[0].shape == (2, 7, 24) and memory[1].shape == (2, 2, 24)
    assert torch.equal(memory[3], pose[:, 0])


@pytest.mark.parametrize("cell", CELLS)
def test_padding_invariance(cell):
    """Padded keyframe slots change neither the state nor the logits."""
    m = RecurrentPlanner(_cfg(cell)).eval()
    tok, pose, valid, query, goal = _batch(B=1, K=5)
    ref = m(tok[valid], pose, valid, query, goal)
    tok_p = torch.cat([tok, torch.randn(1, 3, 5, 384)], 1)
    pose_p = torch.cat([pose, torch.randn(1, 3, 3)], 1)
    valid_p = torch.cat([valid, torch.zeros(1, 3, dtype=torch.bool)], 1)
    assert torch.allclose(ref, m(tok_p[valid_p], pose_p, valid_p, query, goal), atol=1e-5)


@pytest.mark.parametrize("cell", CELLS)
def test_all_params_get_grad(cell):
    m = RecurrentPlanner(_cfg(cell)).train()
    tok, pose, valid, query, goal = _batch()
    m(tok[valid], pose, valid, query, goal).sum().backward()
    assert not [n for n, p in m.named_parameters() if p.grad is None]


@pytest.mark.parametrize("cell", CELLS)
def test_order_sensitive(cell):
    """A recurrent memory depends on keyframe order (the spatial Planner does not)."""
    m = RecurrentPlanner(_cfg(cell)).eval()
    tok, pose, valid, query, goal = _batch(B=1, K=6)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    a = m.encode_map(m.summarize(tok[valid], valid), pose, valid, goal)
    b = m.encode_map(m.summarize(tok[:, perm][valid], valid), pose[:, perm], valid, goal)
    assert not torch.allclose(a[1], b[1], atol=1e-4)


def test_encode_read_matches_forward():
    m = RecurrentPlanner(_cfg("sru")).eval()
    tok, pose, valid, query, goal = _batch()
    memory = m.encode_map(m.summarize(tok[valid], valid), pose, valid, goal)
    assert torch.allclose(m.read(memory, query, goal), m(tok[valid], pose, valid, query, goal), atol=1e-6)


def test_decoder_is_autoregressive():
    """The chunk decoder conditions slot k on the action at slot k-1: teacher actions leave slot 0
    unchanged and change the later slots."""
    m = RecurrentPlanner(_cfg("lstm")).eval()
    nn.init.normal_(m.decoder.out.weight)
    tok, pose, valid, query, goal = _batch()
    memory = m.encode_map(m.summarize(tok[valid], valid), pose, valid, goal)
    greedy = m.read(memory, query, goal)
    forced = m.read(memory, query, goal, teacher_actions=(greedy.argmax(-1) + 1) % N_ACTIONS)
    assert torch.allclose(greedy[:, :, 0], forced[:, :, 0], atol=1e-6)
    assert not torch.allclose(greedy[:, :, 1:], forced[:, :, 1:], atol=1e-3)


@pytest.mark.parametrize("cell", CELLS)
def test_query_step_reads_h_and_c(cell):
    """A query is one recurrent step from the tour's final state, so both h and c reach the logits."""
    m = RecurrentPlanner(_cfg(cell)).eval()
    nn.init.normal_(m.decoder.out.weight)
    tok, pose, valid, query, goal = _batch()
    actions = torch.randint(0, N_ACTIONS, (2, 3, 8))
    with torch.no_grad():
        tokens, h, c, anchor = m.encode_map(m.summarize(tok[valid], valid), pose, valid, goal)
        ref = m.read((tokens, h, c, anchor), query, goal, actions)
        for memory in ((tokens, h + 0.5, c, anchor), (tokens, h, c + 0.5, anchor)):
            assert not torch.allclose(ref, m.read(memory, query, goal, actions), atol=1e-3)


def _naive_sru(stack, x, valid):
    """Per-step, per-layer SRUCell loop: the reference for the layer-major recurrence."""
    B, K, _ = x.shape
    hs = [x.new_zeros(B, stack.hidden) for _ in range(stack.layers)]
    cs = [x.new_zeros(B, stack.hidden) for _ in range(stack.layers)]
    outs = []
    for t in range(K):
        xt = x[:, t]
        for layer, cell in enumerate(stack.cells):
            nh, nc = cell(xt, hs[layer], cs[layer])
            m = valid[:, t].unsqueeze(-1)
            hs[layer], cs[layer] = torch.where(m, nh, hs[layer]), torch.where(m, nc, cs[layer])
            xt = hs[layer]
        outs.append(hs[-1])
    return torch.stack(outs, 1), torch.stack(hs), torch.stack(cs)


@pytest.mark.parametrize("cell", ("sru", "sru_gate"))
def test_sru_layer_major_matches_naive(cell):
    torch.manual_seed(0)
    stack = SRUStack(10, 12, layers=2, refine_gate=cell == "sru_gate").eval()
    x = torch.randn(3, 9, 10)
    valid = torch.ones(3, 9, dtype=torch.bool)
    valid[1, 6:] = False
    valid[2, 3:] = False
    outs, (h, c) = stack(x, valid)
    ref_outs, ref_h, ref_c = _naive_sru(stack, x, valid)
    assert torch.allclose(outs, ref_outs, atol=1e-6)
    assert torch.allclose(h, ref_h, atol=1e-6) and torch.allclose(c, ref_c, atol=1e-6)


@pytest.mark.parametrize("cell", ("sru", "sru_gate"))
def test_sru_custom_bptt_gradients(cell):
    """Manual BPTT reproduces autograd through the naive loop, padded steps included (float64)."""
    torch.manual_seed(1)
    stack = SRUStack(7, 9, layers=2, refine_gate=cell == "sru_gate").double()
    x = torch.randn(3, 8, 7, dtype=torch.float64, requires_grad=True)
    valid = torch.ones(3, 8, dtype=torch.bool)
    valid[1, 5:] = False
    w = torch.randn(3, 8, 9, dtype=torch.float64)

    def loss(outs, h, c):
        return (outs * w).sum() + h.sum() + 0.5 * c.sum()

    outs, (h, c) = stack(x, valid)
    loss(outs, h, c).backward()
    fast = [p.grad.clone() for p in stack.parameters()] + [x.grad.clone()]
    stack.zero_grad()
    x.grad = None
    loss(*_naive_sru(stack, x, valid)).backward()
    naive = [p.grad.clone() for p in stack.parameters()] + [x.grad.clone()]
    for a, b in zip(fast, naive):
        assert torch.allclose(a, b, atol=1e-9, rtol=1e-7), (a - b).abs().max()


@pytest.mark.parametrize("cell", CELLS)
def test_goal_rides_every_step(cell):
    """The goal enters every recurrent step, seen from each keyframe's own pose, so the tour
    state depends on it."""
    m = RecurrentPlanner(_cfg(cell)).eval()
    tok, pose, valid, query, goal = _batch(B=1, K=5)
    e = m.summarize(tok[valid], valid)
    a = m.encode_map(e, pose, valid, goal)
    b = m.encode_map(e, pose, valid, goal + 1.0)
    assert not torch.allclose(a[1], b[1], atol=1e-6), "the tour state is goal-agnostic"
    gf = goal_features(pose, goal[:, None].expand(1, 5, 2), 10.0)
    assert gf.shape == (1, 5, 3) and not torch.allclose(gf[0, 0], gf[0, -1], atol=1e-6)


@pytest.mark.parametrize("arch", ("planner",) + CELLS)
def test_world_axes_like_the_planner(arch):
    """Both models compare positions in world axes: moving the tour, queries and goal together
    leaves the output unchanged, rotating them together does not."""
    torch.manual_seed(0)
    if arch == "planner":
        m = Planner(PlannerConfig(summary_dim=32, model_dim=48, num_heads=2, map_layers=1, read_layers=1))
    else:
        m = RecurrentPlanner(_cfg(arch))
    m.eval()
    nn.init.normal_(m.decoder.out.weight)          # O(1) logits, so the tolerances bite
    tok, pose, valid, query, goal = _batch(B=1, K=5)
    actions = torch.randint(0, N_ACTIONS, (1, 3, 8))

    def run(th, shift):
        R = torch.tensor([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
        p, q, g = pose.clone(), query.clone(), goal @ R.T + shift
        for x in (p, q):
            x[..., :2] = x[..., :2] @ R.T + shift
            x[..., 2] += th
        with torch.no_grad():
            return m(tok[valid], p, valid, q, g, teacher_actions=actions)

    ref = run(0.0, torch.zeros(2))
    assert torch.allclose(ref, run(0.0, torch.tensor([3.0, -2.0])), atol=1e-4)
    assert not torch.allclose(ref, run(0.7, torch.zeros(2)), atol=1e-3)


@pytest.mark.parametrize("cell", CELLS)
def test_packed_summarize_scatters_rows(cell):
    m = RecurrentPlanner(_cfg(cell)).eval()
    tok, pose, valid, query, goal = _batch(B=2, K=6)
    valid[1, 4:] = False
    packed = tok[valid]
    e = m.summarize(packed, valid)
    for i, n in enumerate((6, 4)):
        solo = m.summarize(tok[i, :n], torch.ones(1, n, dtype=torch.bool))
        assert torch.allclose(solo[0], e[i, :n], atol=1e-6) and torch.all(e[i, n:] == 0)
    memory = m.encode_map(e, pose, valid, goal)
    assert torch.allclose(m.read(memory, query, goal), m(packed, pose, valid, query, goal), atol=1e-6)


@pytest.mark.parametrize("cell", CELLS)
def test_motion_input_is_the_only_adjacency_input(cell):
    """Moving keyframe 3 changes the recurrence's input at step 3 only; with motion_input on, step 4's motion
    from keyframe 3 changes too. The query step runs at the same input width."""
    tok, pose, valid, query, goal = _batch(B=1, K=6)
    moved = pose.clone()
    moved[:, 3, :2] += 1.0
    for motion_input, expected in ((False, [3]), (True, [3, 4])):
        torch.manual_seed(0)
        m = RecurrentPlanner(dataclasses.replace(_cfg(cell), motion_input=motion_input)).eval()
        inputs = []
        handle = m.rnn.register_forward_hook(lambda mod, args, out: inputs.append(args[0].detach()))
        with torch.no_grad():
            e = m.summarize(tok[valid], valid)
            memory = m.encode_map(e, pose, valid, goal)
            m.encode_map(e, moved, valid, goal)
            m.read(memory, query, goal)
        handle.remove()
        assert inputs[0].shape[-1] == 32 + FRAME_FEATURES + (MOTION_FEATURES if motion_input else 0)
        changed = torch.nonzero((inputs[0] - inputs[1]).abs().amax(-1)[0] > 1e-6).flatten().tolist()
        assert changed == expected, (cell, motion_input, changed)
