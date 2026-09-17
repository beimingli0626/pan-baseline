"""Classical baseline: OctoMap scan-insertion semantics, the camera frame, the squash bands, the footprint,
A* against Dijkstra, line of sight, and the policy on a synthetic map."""
import math

import numpy as np
import pytest
import quaternion as npq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from baselines.classical import (AGENT_HEIGHT, NAVMESH_MAX_CLIMB, ClassicalConfig, GroundMap, OccupancyAStarPolicy,
                                 OccupancyGrid, _move_ok, astar, camera_to_world, farthest_visible, footprint,
                                 insert_scan, line_of_sight, squash, traversable)
from pan.util.constants import FORWARD_STEP_M, Action
from pan.util.geometry import heading_from_rotation, wrap_angle

RES = 0.05
HIT, MISS, LO, HI = 0.85, -0.4, -2.0, 3.5


def _grid(n=20):
    return np.zeros((n, n, n), np.float32), np.zeros((n, n, n), np.bool_), np.zeros((n, n, n), np.int32)


def _insert(grid, scan, origin, points, max_range=-1.0):
    logodds, known, stamp = grid
    insert_scan(logodds, known, stamp, scan, np.asarray(origin, np.float64), np.asarray(points, np.float64),
                np.zeros(3, np.int64), RES, max_range, HIT, MISS, LO, HI)


def _segment_voxels(o, e, n):
    """Voxels whose cube the segment o -> e crosses over a positive length (slab clipping)."""
    hits = set()
    d = e - o
    for key in np.ndindex(n, n, n):
        lo, hi = np.array(key) * RES, (np.array(key) + 1) * RES
        t0, t1 = 0.0, 1.0
        for a in range(3):
            if abs(d[a]) < 1e-12:
                if not lo[a] <= o[a] < hi[a]:
                    t0, t1 = 1.0, 0.0
                continue
            ta, tb = sorted(((lo[a] - o[a]) / d[a], (hi[a] - o[a]) / d[a]))
            t0, t1 = max(t0, ta), min(t1, tb)
        if t1 - t0 > 1e-9:
            hits.add(key)
    return hits


def test_ray_traversal_is_the_segment_minus_its_endpoint_voxel():
    rng = np.random.default_rng(0)
    for _ in range(25):
        o, e = rng.uniform(0.0, 20 * RES, 3), rng.uniform(0.0, 20 * RES, 3)
        grid = _grid()
        _insert(grid, 0, o, e[None])
        logodds, known, _ = grid
        end = tuple(np.floor(e / RES).astype(int))
        expected = _segment_voxels(o, e, 20) - {end}
        free = {tuple(k) for k in np.argwhere(known & (logodds < 0))}
        assert free == expected
        assert logodds[end] == np.float32(HIT) and known.sum() == len(expected) + 1


def test_one_update_per_scan_and_hits_win():
    grid = _grid()
    origin = [0.5 * RES, 0.5 * RES, 0.5 * RES]
    points = [[2.5 * RES, 0.5 * RES, 0.5 * RES], [5.5 * RES, 0.5 * RES, 0.5 * RES]]   # the second crosses the first's end
    _insert(grid, 0, origin, points)
    row = grid[0][:7, 0, 0]
    np.testing.assert_allclose(row, [MISS, MISS, HIT, MISS, MISS, HIT, 0.0], rtol=1e-6)
    for scan in range(1, 10):
        _insert(grid, scan, origin, points)
    np.testing.assert_allclose(grid[0][:6, 0, 0], [LO, LO, HI, LO, LO, HI], rtol=1e-6)


def test_max_range_clears_without_a_hit():
    grid = _grid()
    _insert(grid, 0, [0.5 * RES, 0.5 * RES, 0.5 * RES], [[5.5 * RES, 0.5 * RES, 0.5 * RES]], max_range=4.2 * RES)
    np.testing.assert_allclose(grid[0][:6, 0, 0], [MISS] * 4 + [0.0, 0.0], rtol=1e-6)
    assert grid[1][:6, 0, 0].tolist() == [True] * 4 + [False, False]


def test_camera_to_world_is_the_habitat_agent_frame():
    rng = np.random.default_rng(3)
    pose = np.stack([rng.uniform(-5, 5, 8), rng.uniform(-5, 5, 8), rng.uniform(-np.pi, np.pi, 8)], 1)
    T = camera_to_world(pose, np.full(8, 0.6))
    for p, t in zip(pose, T):
        agent_rotation = npq.quaternion(math.cos(p[2] / 2), 0.0, math.sin(p[2] / 2), 0.0)   # env.Stepper.reset
        np.testing.assert_allclose(t[:3, :3], npq.as_rotation_matrix(agent_rotation), atol=1e-12)
        assert abs(wrap_angle(heading_from_rotation(t) - p[2])) < 1e-9
        np.testing.assert_allclose(t[:3, 3], [p[0], 0.6, p[1]])


def test_squash_bands():
    ny = int(round((NAVMESH_MAX_CLIMB + AGENT_HEIGHT + 0.3) / RES))
    logodds = np.full((5, ny, 1), -1.0, np.float32)
    known = np.zeros((5, ny, 1), np.bool_)
    floor = 0.0
    key0 = np.array([0, int(math.floor((floor - NAVMESH_MAX_CLIMB) / RES)), 0])
    for column, height in enumerate([0.02, 0.15, 0.32, 0.68, 0.85]):
        j = int(math.floor(height / RES)) - key0[1]
        logodds[column, j, 0], known[column, j, 0] = 2.0, True
    ground = squash(OccupancyGrid(logodds, known, key0, RES), np.full((5, 1), floor))
    assert ground.floor[:, 0].tolist() == [True, True, False, False, False]
    assert ground.obstacle[:, 0].tolist() == [False, False, True, True, False]
    assert ground.observed[:, 0].tolist() == [True, True, True, True, False]


def test_footprint_is_the_disk_square_intersection():
    k = footprint(0.2, RES)
    assert k.shape == (9, 9) and k[4, 4] and k[0, 4] and k[0, 2] and not k[0, 1] and not k[0, 0]


def test_astar_matches_dijkstra():
    rng = np.random.default_rng(1)
    for _ in range(10):
        trav = rng.random((25, 25)) > 0.3
        nx, nz = trav.shape
        src, dst, w = [], [], []
        for i in range(nx):
            for j in range(nz):
                if not trav[i, j]:
                    continue
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if (di or dj) and _move_ok(trav, i, j, di, dj):
                            src.append(i * nz + j)
                            dst.append((i + di) * nz + j + dj)
                            w.append(math.sqrt(2.0) if di and dj else 1.0)
        free = np.argwhere(trav)
        s, g = free[rng.integers(len(free))], free[rng.integers(len(free))]
        dist = dijkstra(csr_matrix((w, (src, dst)), shape=(nx * nz, nx * nz)), indices=s[0] * nz + s[1])
        path = astar(trav, s[0], s[1], g[0], g[1])
        if not np.isfinite(dist[g[0] * nz + g[1]]):
            assert len(path) == 0
            continue
        steps = np.diff(path, axis=0)
        assert all(_move_ok(trav, *a, *b) for a, b in zip(path[:-1], steps))
        cost = sum(math.sqrt(2.0) if abs(a) + abs(b) == 2 else 1.0 for a, b in steps)
        assert abs(cost - dist[g[0] * nz + g[1]]) < 1e-9
        assert tuple(path[0]) == tuple(s) and tuple(path[-1]) == tuple(g)


def test_line_of_sight_and_farthest_visible():
    trav = np.ones((10, 10), np.bool_)
    trav[5, 0:8] = False                                   # a wall across x = 5 with a gap at z >= 8
    assert line_of_sight(trav, 1.5, 1.5, 4.5, 7.5)
    assert not line_of_sight(trav, 1.5, 1.5, 8.5, 1.5)
    assert line_of_sight(trav, 1.5, 8.5, 8.5, 8.5)
    vertices = np.array([[2.5, 5.5], [4.5, 8.5], [6.5, 8.5], [8.5, 1.5]])
    assert farthest_visible(trav, 1.5, 1.5, vertices, 1.0) == 1


def _gap_world():
    """3 m x 3 m, a wall at x in [1.5, 1.6) with a gap at z > 2.2."""
    n = 60
    obstacle = np.zeros((n, n), bool)
    obstacle[30:32, :44] = True
    ground = GroundMap(obstacle, ~obstacle, np.ones((n, n), bool), np.zeros((n, n), np.float32),
                       np.zeros(2, np.int64), RES)
    return ground, obstacle


def _drive(policy, obstacle, goal, steps=300):
    """Exact discrete kinematics without collisions -> (last action, final x, final z)."""
    x, z, yaw = 0.5, 0.5, 0.0
    for _ in range(steps):
        action = policy(np.array([x, z, yaw]), goal)
        if action == Action.STOP:
            break
        if action == Action.FORWARD:
            x, z = x - math.sin(yaw) * FORWARD_STEP_M, z - math.cos(yaw) * FORWARD_STEP_M
            assert not obstacle[int(x / RES), int(z / RES)]
        else:
            yaw = wrap_angle(yaw + math.radians(30.0) * (1 if action == Action.TURN_LEFT else -1))
    return action, x, z


@pytest.mark.parametrize("exec_horizon", [1, 4])
def test_policy_goes_through_the_gap_and_stops(exec_horizon):
    ground, obstacle = _gap_world()
    cfg = ClassicalConfig()
    policy = OccupancyAStarPolicy(ground, traversable(ground, cfg), cfg, exec_horizon=exec_horizon)
    goal = np.array([2.5, 0.5])
    action, x, z = _drive(policy, obstacle, goal)
    assert action == Action.STOP and math.hypot(x - goal[0], z - goal[1]) <= 0.15
    assert policy.n_fallback == 0 and policy.first_plan[:, 1].max() > 2.2 and policy.n_blocked == 0


def test_band_top_decides_overhangs():
    ny = int(round((NAVMESH_MAX_CLIMB + 1.0) / RES))
    logodds = np.full((1, ny, 1), -1.0, np.float32)
    known = np.zeros((1, ny, 1), np.bool_)
    key0 = np.array([0, int(math.floor(-NAVMESH_MAX_CLIMB / RES)), 0])
    j = int(math.floor(0.76 / RES)) - key0[1]                  # a table top at 0.76 m
    logodds[0, j, 0], known[0, j, 0] = 2.0, True
    grid = OccupancyGrid(logodds, known, key0, RES)
    assert not squash(grid, np.zeros((1, 1)), AGENT_HEIGHT).obstacle[0, 0]
    assert squash(grid, np.zeros((1, 1)), 0.8).obstacle[0, 0]


def test_observed_first_detours_around_unobserved_cells():
    ground, obstacle = _gap_world()
    observed = np.ones_like(obstacle)
    observed[20:40, 4:30] = False                            # the straight way to the goal, never seen
    ground = GroundMap(obstacle, ~obstacle, observed, ground.p_obstacle, ground.key0, RES)
    obstacle[:] = False                                      # no wall: only the unobserved patch is in the way
    ground.obstacle[:] = False
    goal = np.array([2.5, 0.5])
    plans = {}
    for observed_first in (False, True):
        cfg = ClassicalConfig(observed_first=observed_first)
        policy = OccupancyAStarPolicy(ground, traversable(ground, cfg), cfg)
        action, x, z = _drive(policy, obstacle, goal)
        assert action == Action.STOP and math.hypot(x - goal[0], z - goal[1]) <= 0.15
        cells = np.floor(policy.first_plan / RES).astype(int)
        plans[observed_first] = (~observed[cells[:, 0], cells[:, 1]]).any()
    assert plans[False] and not plans[True]


def test_bump_marking_needs_exact_per_step_poses():
    ground, _ = _gap_world()
    cfg = ClassicalConfig(bump_marking=True)
    for kwargs in (dict(exec_horizon=4), dict(query_noise=(0.1, 0.2), rng=np.random.default_rng(0))):
        with pytest.raises(AssertionError):
            OccupancyAStarPolicy(ground, traversable(ground, cfg), cfg, **kwargs)
