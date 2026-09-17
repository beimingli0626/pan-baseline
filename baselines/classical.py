"""Classical baseline: an OctoMap occupancy map of the tour's depth frames, squashed to a 2-D grid
over the robot's height band, and A* on that grid.

Mapping. OctoMap's scan insertion (Hornung et al., Autonomous Robots 2013) on a dense voxel grid
with OctoMap's key layout (voxel k spans [k res, (k + 1) res)). Every valid depth pixel casts a ray
from the camera centre: the voxels it traverses (OctoMap's computeRayKeys, origin voxel included,
endpoint voxel excluded) take a miss and its endpoint voxel a hit, each voxel at most one update
per frame with hits taking precedence, log-odds clamped. A leaf of OctoMap's pruned octree holds
the value of the dense voxel, so the occupancy probabilities are OctoMap's. The grid spans every
ray endpoint in x and z, and in height only the two bands below.

Squashing. A ground cell's floor is the floor under the nearest context camera (the camera's
height minus SENSOR_HEIGHT). An occupied voxel (P >= 0.5) more than NAVMESH_MAX_CLIMB and at most
`band_top` above that floor makes the cell an obstacle; one within NAVMESH_MAX_CLIMB of it is
observed floor. A cell is blocked when the disk of AGENT_RADIUS around its centre meets the square
of an obstacle cell.

Planning and control. 8-connected A* (no corner cutting, octile heuristic) from the agent's cell to
the goal's, through observed cells first when `observed_first`; when the goal cell is blocked or
cut off, to the reachable cell nearest the goal and then straight at the goal. The agent steers at
the farthest path vertex in line of sight and acts as the expert does: STOP within
`expert_stop_radius` of the goal, turn when the heading error exceeds half a turn step, else
FORWARD. As in the learned planners' chunk commit, a query reads the agent's pose once and commits
`exec_horizon` actions, each planned at the pose the previous ones lead to on the discrete
kinematics.

Inputs are the learned planners' evaluation inputs: the context frames' depth and camera poses
(under the store's keyframe noise), the agent's pose at each query (under the query noise) and the
goal position; the navmesh only moves and scores the agent. The optional bump marking (a FORWARD
that moves less than half a step blocks the cell ahead for the rest of the episode) reads
consecutive exact agent poses, an online collision signal the learned planners do not have.

Usage (from the repository root, PYTHONPATH=src:baselines):
    python -m baselines.classical --token-root <val_tokens> --traj-root <val_shards> \
        --bank <bank.npz> --eta 10 --exec-horizon 4 --workers 16 --out logs/<run>/eval/<tag>.json
"""
from __future__ import annotations

import argparse
import dataclasses
import heapq
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import numpy as np
from numba import njit
from scipy.ndimage import binary_dilation, distance_transform_edt

from pan.planner.dataset import DataConfig, EvalBank, TourStore, jitter_
from pan.planner.env import EpisodeConfig, Stepper, load_pathfinder, policy_rollout
from pan.planner.eval import row_result, summarize_results, write_json
from pan.util.config import add_config_args, arg, config_from_args
from pan.util.constants import AGENT_HEIGHT, AGENT_RADIUS, FORWARD_STEP_M, SENSOR_HEIGHT, SENSOR_RES, Action
from pan.util.geometry import CX, CY, DEPTH_MIN, FX, FY, heading_of, wrap_angle

NAVMESH_MAX_CLIMB = 0.2         # habitat NavMeshSettings.agent_max_climb default, which env.bake_navmesh keeps
NAVMESH_CELL_HEIGHT = 0.2       # habitat NavMeshSettings.cell_height default: the bake's vertical voxel
OCCUPIED_LOGODDS = 0.0          # OctoMap's occupancy threshold: occupied at log-odds >= 0 (P >= 0.5)
SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class ClassicalConfig:
    resolution: float = arg(0.05, "m, voxel edge and ground-cell edge")
    max_range: float = arg(-1.0, "m; a longer ray only clears space up to this range (-1: unbounded, "
                                 "OctoMap's default)")
    hit: float = arg(0.85, "log-odds a hit adds (OctoMap's default, P 0.70)")
    miss: float = arg(-0.4, "log-odds a miss adds (OctoMap's default, P 0.40)")
    clamp_min: float = arg(-2.0, "lower log-odds clamp (OctoMap's default, P 0.12)")
    clamp_max: float = arg(3.5, "upper log-odds clamp (OctoMap's default, P 0.97)")
    band_top: float = arg(AGENT_HEIGHT, "m above the floor: occupied voxels up to this height are obstacles (the "
                                        "agent's height; the navmesh bake rounds its clearance up to whole cells, "
                                        "ceil(AGENT_HEIGHT / NAVMESH_CELL_HEIGHT) * NAVMESH_CELL_HEIGHT = 0.8 m)")
    floor_required: bool = arg(False, "plan only through cells with observed floor; by default every "
                                      "unblocked cell is traversable, observed or not")
    observed_first: bool = arg(True, "plan through observed cells whenever such a route to the goal exists, through "
                                     "unobserved cells only otherwise (off: every unblocked cell alike, observed or not)")
    bump_marking: bool = arg(False, "online collision memory: a FORWARD that moves less than half a step "
                                    "marks the cell ahead as an obstacle for the rest of the episode (reads "
                                    "consecutive exact agent poses, which the learned planners do not)")


# ---------------------------------------------------------------- OctoMap scan insertion
@njit(cache=True)
def _apply(logodds, known, stamp, i, j, k, tag, delta, lo, hi):
    v = delta + logodds[i, j, k] if known[i, j, k] else delta
    logodds[i, j, k] = min(max(v, lo), hi)
    known[i, j, k] = True
    stamp[i, j, k] = tag


@njit(cache=True)
def _clear(logodds, known, stamp, occ_tag, free_tag, key0, ki, kj, kk, miss, lo, hi):
    i, j, k = ki - key0[0], kj - key0[1], kk - key0[2]
    nx, ny, nz = logodds.shape
    if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
        s = stamp[i, j, k]
        if s != occ_tag and s != free_tag:
            _apply(logodds, known, stamp, i, j, k, free_tag, miss, lo, hi)


@njit(cache=True)
def insert_scan(logodds, known, stamp, scan, origin, points, key0, res, max_range, hit, miss, lo, hi):
    """One OctoMap insertPointCloud of `points` (N, 3) seen from `origin` (3,) into the voxels
    (logodds, known) with keys key0 + index. `stamp` (int32, like the grid) holds each voxel's last
    update tag: scan s tags hits 2s + 1 and misses 2s + 2, so a voxel takes one update per scan and a
    hit wins over a miss."""
    nx, ny, nz = logodds.shape
    occ_tag, free_tag = 2 * scan + 1, 2 * scan + 2
    for p in range(points.shape[0]):
        d0, d1, d2 = points[p, 0] - origin[0], points[p, 1] - origin[1], points[p, 2] - origin[2]
        if max_range >= 0.0 and math.sqrt(d0 * d0 + d1 * d1 + d2 * d2) > max_range:
            continue
        i = int(math.floor(points[p, 0] / res)) - key0[0]
        j = int(math.floor(points[p, 1] / res)) - key0[1]
        k = int(math.floor(points[p, 2] / res)) - key0[2]
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and stamp[i, j, k] != occ_tag:
            _apply(logodds, known, stamp, i, j, k, occ_tag, hit, lo, hi)
    o0 = int(math.floor(origin[0] / res))
    o1 = int(math.floor(origin[1] / res))
    o2 = int(math.floor(origin[2] / res))
    inf = np.inf
    for p in range(points.shape[0]):
        d0, d1, d2 = points[p, 0] - origin[0], points[p, 1] - origin[1], points[p, 2] - origin[2]
        length = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        if length == 0.0:
            continue
        if max_range >= 0.0 and length > max_range:        # clear up to max_range, no hit
            d0, d1, d2, length = d0 * max_range / length, d1 * max_range / length, d2 * max_range / length, max_range
        e0 = int(math.floor((origin[0] + d0) / res))
        e1 = int(math.floor((origin[1] + d1) / res))
        e2 = int(math.floor((origin[2] + d2) / res))
        if e0 == o0 and e1 == o1 and e2 == o2:
            continue
        _clear(logodds, known, stamp, occ_tag, free_tag, key0, o0, o1, o2, miss, lo, hi)
        # computeRayKeys: Amanatides-Woo traversal from the origin voxel
        u0, u1, u2 = d0 / length, d1 / length, d2 / length
        s0 = 1 if u0 > 0.0 else (-1 if u0 < 0.0 else 0)
        s1 = 1 if u1 > 0.0 else (-1 if u1 < 0.0 else 0)
        s2 = 1 if u2 > 0.0 else (-1 if u2 < 0.0 else 0)
        t0 = ((o0 + 0.5) * res + s0 * res * 0.5 - origin[0]) / u0 if s0 != 0 else inf
        t1 = ((o1 + 0.5) * res + s1 * res * 0.5 - origin[1]) / u1 if s1 != 0 else inf
        t2 = ((o2 + 0.5) * res + s2 * res * 0.5 - origin[2]) / u2 if s2 != 0 else inf
        dt0 = res / abs(u0) if s0 != 0 else inf
        dt1 = res / abs(u1) if s1 != 0 else inf
        dt2 = res / abs(u2) if s2 != 0 else inf
        c0, c1, c2 = o0, o1, o2
        while True:
            if t0 < t1:
                dim = 0 if t0 < t2 else 2
            else:
                dim = 1 if t1 < t2 else 2
            if dim == 0:
                c0 += s0
                t0 += dt0
            elif dim == 1:
                c1 += s1
                t1 += dt1
            else:
                c2 += s2
                t2 += dt2
            if c0 == e0 and c1 == e1 and c2 == e2:
                break
            if min(t0, min(t1, t2)) > length:              # numerical overshoot of the endpoint
                break
            _clear(logodds, known, stamp, occ_tag, free_tag, key0, c0, c1, c2, miss, lo, hi)


# ---------------------------------------------------------------- the 3-D map
_U, _V = np.meshgrid(np.arange(SENSOR_RES[1], dtype=np.float64), np.arange(SENSOR_RES[0], dtype=np.float64))
RAY_X, RAY_Y = (_U - CX) / FX, -(_V - CY) / FY          # camera-frame ray per metre of depth


def frame_points(depth: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    """(H, W) metric depth + its camera-to-world -> (N, 3) world points of its valid pixels."""
    d = np.asarray(depth, np.float64)
    valid = d > DEPTH_MIN
    dv = d[valid]
    p_camera = np.stack([RAY_X[valid] * dv, RAY_Y[valid] * dv, -dv], 1)
    return p_camera @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]


def R_y(yaw: np.ndarray) -> np.ndarray:
    """(...,) yaw -> (..., 3, 3) rotation about +y; at yaw an agent faces (-sin yaw, 0, -cos yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros(np.shape(yaw) + (3, 3))
    R[..., 0, 0], R[..., 0, 2], R[..., 1, 1], R[..., 2, 0], R[..., 2, 2] = c, s, 1.0, -s, c
    return R


def camera_to_world(pose: np.ndarray, camera_height: np.ndarray) -> np.ndarray:
    """(K, 3) agent poses (x, z, yaw) + (K,) camera heights -> (K, 4, 4) T_world_camera of the upright
    camera above each agent (the sensor offset is a pure lift, so the camera shares the agent's yaw)."""
    T = np.zeros((len(pose), 4, 4))
    T[:, :3, :3] = R_y(pose[:, 2])
    T[:, :3, 3] = np.stack([pose[:, 0], camera_height, pose[:, 1]], 1)
    T[:, 3, 3] = 1.0
    return T


@dataclass
class OccupancyGrid:
    """Log-odds voxels: voxel (i, j, k) has OctoMap key key0 + (i, j, k); `known` marks the voxels any
    ray updated (the leaves OctoMap would hold)."""
    logodds: np.ndarray             # (nx, ny, nz) float32
    known: np.ndarray               # (nx, ny, nz) bool
    key0: np.ndarray                # (3,) int64
    res: float

    def probability(self) -> np.ndarray:
        return np.where(self.known, 1.0 / (1.0 + np.exp(-self.logodds)), np.nan).astype(np.float32)


def build_map(depths: np.ndarray, T_world_cameras: np.ndarray, cfg: ClassicalConfig) -> Tuple[OccupancyGrid, np.ndarray]:
    """OctoMap of the frames, cropped in height to the bands of `squash` -> (grid, floor_ref (nx, nz),
    the floor height under the nearest camera)."""
    res = cfg.resolution
    floor_y = T_world_cameras[:, 1, 3] - SENSOR_HEIGHT
    lo = T_world_cameras[:, [0, 2], 3].min(0)
    hi = T_world_cameras[:, [0, 2], 3].max(0)
    for depth, T in zip(depths, T_world_cameras):         # x / z bounds: every ray end
        points = frame_points(depth, T)
        if cfg.max_range >= 0.0 and len(points):
            ray = points - T[:3, 3]
            length = np.linalg.norm(ray, axis=1, keepdims=True)
            points = T[:3, 3] + ray * np.minimum(1.0, cfg.max_range / np.maximum(length, 1e-9))
        if len(points):
            lo = np.minimum(lo, points[:, [0, 2]].min(0))
            hi = np.maximum(hi, points[:, [0, 2]].max(0))
    key_lo = np.floor(np.array([lo[0], floor_y.min() - NAVMESH_MAX_CLIMB, lo[1]]) / res).astype(np.int64)
    key_hi = np.floor(np.array([hi[0], floor_y.max() + cfg.band_top, hi[1]]) / res).astype(np.int64)
    shape = tuple(int(n) for n in key_hi - key_lo + 1)
    logodds = np.zeros(shape, np.float32)
    known = np.zeros(shape, np.bool_)
    stamp = np.zeros(shape, np.int32)
    for s, (depth, T) in enumerate(zip(depths, T_world_cameras)):
        insert_scan(logodds, known, stamp, s, np.ascontiguousarray(T[:3, 3], np.float64), frame_points(depth, T),
                    key_lo, res, cfg.max_range, cfg.hit, cfg.miss, cfg.clamp_min, cfg.clamp_max)
    seeds = np.zeros((shape[0], shape[2]), bool)
    floor_seed = np.zeros((shape[0], shape[2]), np.float64)
    ci = np.floor(T_world_cameras[:, 0, 3] / res).astype(np.int64) - key_lo[0]
    cj = np.floor(T_world_cameras[:, 2, 3] / res).astype(np.int64) - key_lo[2]
    seeds[ci, cj], floor_seed[ci, cj] = True, floor_y
    nearest = distance_transform_edt(~seeds, return_distances=False, return_indices=True)
    return OccupancyGrid(logodds, known, key_lo, res), floor_seed[nearest[0], nearest[1]]


# ---------------------------------------------------------------- the 2-D map
@dataclass
class GroundMap:
    """The squashed map on the ground cells (nx, nz) of an OccupancyGrid."""
    obstacle: np.ndarray            # bool: an occupied voxel in (floor + climb, floor + band_top]
    floor: np.ndarray               # bool: an occupied voxel in [floor - climb, floor + climb]
    observed: np.ndarray            # bool: any updated voxel in either band
    p_obstacle: np.ndarray          # float32: max P(occupied) over updated obstacle-band voxels, NaN if none
    key0: np.ndarray                # (2,) int64 OctoMap keys of cell (0, 0) in x and z
    res: float


def squash(grid: OccupancyGrid, floor_ref: np.ndarray, band_top: float = AGENT_HEIGHT) -> GroundMap:
    nx, ny, nz = grid.logodds.shape
    obstacle, floor, observed = (np.zeros((nx, nz), bool) for _ in range(3))
    p_obstacle = np.full((nx, nz), np.nan, np.float32)
    for j in range(ny):
        height = (grid.key0[1] + j + 0.5) * grid.res - floor_ref
        in_obstacle = (height > NAVMESH_MAX_CLIMB) & (height <= band_top)
        in_floor = np.abs(height) <= NAVMESH_MAX_CLIMB
        known = grid.known[:, j]
        occupied = known & (grid.logodds[:, j] >= OCCUPIED_LOGODDS)
        obstacle |= occupied & in_obstacle
        floor |= occupied & in_floor
        observed |= known & (in_obstacle | in_floor)
        p = np.where(known & in_obstacle, 1.0 / (1.0 + np.exp(-grid.logodds[:, j])), np.nan)
        p_obstacle = np.fmax(p_obstacle, p.astype(np.float32))
    return GroundMap(obstacle, floor, observed, p_obstacle, grid.key0[[0, 2]].copy(), grid.res)


def footprint(radius: float, res: float) -> np.ndarray:
    """Cell offsets whose square (edge `res`) meets the disk of `radius` around a cell centre."""
    n = int(math.ceil(radius / res + 0.5)) - 1             # the largest |offset| with (|offset| - 0.5) res < radius
    gap = np.maximum(np.abs(np.arange(-n, n + 1)) - 0.5, 0.0) * res
    return gap[:, None] ** 2 + gap[None, :] ** 2 < radius ** 2


def traversable(ground: GroundMap, cfg: ClassicalConfig) -> np.ndarray:
    free = ~binary_dilation(ground.obstacle, structure=footprint(AGENT_RADIUS, ground.res))
    return free & ground.floor if cfg.floor_required else free


def cell_ahead(ground: GroundMap, x: float, z: float, yaw: float) -> Tuple[int, int]:
    """The ground cell a blocked FORWARD ran into: 1.5 cells past the body's front, so its footprint
    never covers the agent's own cell but covers the cell in front of it."""
    reach = AGENT_RADIUS + 1.5 * ground.res
    return (int(math.floor((x - math.sin(yaw) * reach) / ground.res)) - int(ground.key0[0]),
            int(math.floor((z - math.cos(yaw) * reach) / ground.res)) - int(ground.key0[1]))


def cell_state(ground: GroundMap, i: int, j: int) -> str:
    if not (0 <= i < ground.obstacle.shape[0] and 0 <= j < ground.obstacle.shape[1]):
        return "outside"
    return "obstacle" if ground.obstacle[i, j] else ("observed" if ground.observed[i, j] else "unobserved")


# ---------------------------------------------------------------- grid search
@njit(cache=True)
def _move_ok(trav, ui, uj, di, dj):
    nx, nz = trav.shape
    vi, vj = ui + di, uj + dj
    if vi < 0 or vi >= nx or vj < 0 or vj >= nz or not trav[vi, vj]:
        return False
    return di == 0 or dj == 0 or (trav[ui + di, uj] and trav[ui, uj + dj])


@njit(cache=True)
def astar(trav, si, sj, gi, gj):
    """8-connected A* over `trav` without corner cutting, step costs 1 and sqrt 2 (cells), octile
    heuristic -> path cells (M, 2) from (si, sj) to (gi, gj); (0, 2) when unreachable."""
    nx, nz = trav.shape
    g = np.full(nx * nz, np.inf)
    parent = np.full(nx * nz, -1, np.int64)
    closed = np.zeros(nx * nz, np.bool_)
    start, goal = si * nz + sj, gi * nz + gj
    g[start] = 0.0
    a, b = abs(si - gi), abs(sj - gj)
    heap = [(max(a, b) + (SQRT2 - 1.0) * min(a, b), start)]
    while len(heap) > 0:
        _, u = heapq.heappop(heap)
        if u == goal:
            break
        if closed[u]:
            continue
        closed[u] = True
        ui, uj = u // nz, u % nz
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if (di == 0 and dj == 0) or not _move_ok(trav, ui, uj, di, dj):
                    continue
                v = (ui + di) * nz + uj + dj
                if closed[v]:
                    continue
                cost = g[u] + (SQRT2 if di != 0 and dj != 0 else 1.0)
                if cost < g[v]:
                    g[v], parent[v] = cost, u
                    a, b = abs(ui + di - gi), abs(uj + dj - gj)
                    heapq.heappush(heap, (cost + max(a, b) + (SQRT2 - 1.0) * min(a, b), v))
    if not np.isfinite(g[goal]):
        return np.zeros((0, 2), np.int64)
    n, u = 1, goal
    while u != start:
        u, n = parent[u], n + 1
    path = np.empty((n, 2), np.int64)
    u = goal
    for m in range(n - 1, -1, -1):
        path[m, 0], path[m, 1] = u // nz, u % nz
        u = parent[u]
    return path


@njit(cache=True)
def nearest_reachable(trav, si, sj, gx, gz):
    """The cell reachable from (si, sj) under A*'s moves whose centre is nearest (gx, gz) (cell units)."""
    nx, nz = trav.shape
    seen = np.zeros(nx * nz, np.bool_)
    queue = np.empty(nx * nz, np.int64)
    queue[0], seen[si * nz + sj] = si * nz + sj, True
    head, tail = 0, 1
    best, best_d = si * nz + sj, np.inf
    while head < tail:
        u = queue[head]
        head += 1
        ui, uj = u // nz, u % nz
        d = (ui + 0.5 - gx) ** 2 + (uj + 0.5 - gz) ** 2
        if d < best_d:
            best, best_d = u, d
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if (di != 0 or dj != 0) and _move_ok(trav, ui, uj, di, dj):
                    v = (ui + di) * nz + uj + dj
                    if not seen[v]:
                        seen[v] = True
                        queue[tail] = v
                        tail += 1
    return best // nz, best % nz


@njit(cache=True)
def _cell_ok(trav, i, j):
    return 0 <= i < trav.shape[0] and 0 <= j < trav.shape[1] and trav[i, j]


@njit(cache=True)
def line_of_sight(trav, x0, z0, x1, z1):
    """Whether every cell the segment (x0, z0) -> (x1, z1) enters is traversable, the cell of (x0, z0)
    excepted; both side cells count where it passes through a corner (cell units: cell (i, j) spans
    [i, i + 1) x [j, j + 1))."""
    i, j = int(math.floor(x0)), int(math.floor(z0))
    ie, je = int(math.floor(x1)), int(math.floor(z1))
    dx, dz = x1 - x0, z1 - z0
    si = 1 if dx > 0.0 else -1
    sj = 1 if dz > 0.0 else -1
    ti = ((i + (si > 0) - x0) / dx) if dx != 0.0 else np.inf
    tj = ((j + (sj > 0) - z0) / dz) if dz != 0.0 else np.inf
    dti = abs(1.0 / dx) if dx != 0.0 else np.inf
    dtj = abs(1.0 / dz) if dz != 0.0 else np.inf
    while not (i == ie and j == je):
        if min(ti, tj) > 1.0:
            break
        if ti < tj:
            i, ti = i + si, ti + dti
        elif tj < ti:
            j, tj = j + sj, tj + dtj
        else:
            if not (_cell_ok(trav, i + si, j) and _cell_ok(trav, i, j + sj)):
                return False
            i, j, ti, tj = i + si, j + sj, ti + dti, tj + dtj
        if not _cell_ok(trav, i, j):
            return False
    return True


@njit(cache=True)
def farthest_visible(trav, x, z, vertices, lookahead):
    """Index of the farthest of `vertices` (K, 2) in line of sight from (x, z) and at least `lookahead`
    away; else the first at least `lookahead` away; else the last (cell units)."""
    first = -1
    for k in range(vertices.shape[0]):
        if (vertices[k, 0] - x) ** 2 + (vertices[k, 1] - z) ** 2 >= lookahead ** 2:
            first = k
            break
    if first < 0:
        return vertices.shape[0] - 1
    for k in range(vertices.shape[0] - 1, first - 1, -1):
        if line_of_sight(trav, x, z, vertices[k, 0], vertices[k, 1]):
            return k
    return first


# ---------------------------------------------------------------- the policy
class OccupancyAStarPolicy:
    """`policy(pose, goal_position, stepper) -> action` on one tour's traversability grid; one instance
    per episode (bump marks are the episode's own). Each query reads the pose, under `query_noise`
    (metres, radians; independent uniform draws from `rng`, eval_accum's rule), and commits
    `exec_horizon` actions."""

    def __init__(self, ground: GroundMap, trav: np.ndarray, cfg: ClassicalConfig,
                 episode_cfg: EpisodeConfig = EpisodeConfig(), exec_horizon: int = 1,
                 query_noise: Tuple[float, float] = (0.0, 0.0), rng: Optional[np.random.Generator] = None):
        assert not (cfg.bump_marking and (exec_horizon > 1 or any(query_noise))), \
            "bump marking compares consecutive exact agent poses: exec_horizon 1, no query noise"
        self.ground, self.cfg, self.episode_cfg = ground, cfg, episode_cfg
        self.exec_horizon, self.query_noise, self.rng = exec_horizon, query_noise, rng
        self.trav = trav.copy()
        self.trav_observed = trav & ground.observed if cfg.observed_first else None
        self.kernel = footprint(AGENT_RADIUS, ground.res)
        self.queue: List[int] = []
        self._nearest = None                    # nearest-traversable indices, built on demand
        self._last: Optional[Tuple[float, float, int]] = None
        self.n_blocked = self.n_bumps = self.n_fallback = self.n_plans = self.n_unobserved_plans = 0
        self.last_blocked: Optional[Tuple[float, float, float]] = None   # diagnostics: exact pose of the last blocked FORWARD
        self.plan_seconds = 0.0
        self.first_plan: Optional[np.ndarray] = None    # (K, 2) world vertices of the first plan

    def _cells(self, x: float, z: float) -> Tuple[float, float]:
        return x / self.ground.res - self.ground.key0[0], z / self.ground.res - self.ground.key0[1]

    def _world(self, cells: np.ndarray) -> np.ndarray:
        return (cells + self.ground.key0) * self.ground.res

    def _mark_bump(self, x: float, z: float, yaw: float) -> None:
        ci, cj = cell_ahead(self.ground, x, z, yaw)
        r, (nx, nz) = self.kernel.shape[0] // 2, self.trav.shape
        i0, i1, j0, j1 = max(ci - r, 0), min(ci + r + 1, nx), max(cj - r, 0), min(cj + r + 1, nz)
        if i0 < i1 and j0 < j1:
            blocked = self.kernel[i0 - ci + r:i1 - ci + r, j0 - cj + r:j1 - cj + r]
            for grid in (self.trav, self.trav_observed):
                if grid is not None:
                    grid[i0:i1, j0:j1] &= ~blocked
            self._nearest = None
        self.n_bumps += 1

    def _start_cell(self, ax: float, az: float) -> Optional[Tuple[int, int]]:
        nx, nz = self.trav.shape
        i, j = min(max(int(math.floor(ax)), 0), nx - 1), min(max(int(math.floor(az)), 0), nz - 1)
        if self.trav[i, j]:
            return i, j
        if not self.trav.any():
            return None
        if self._nearest is None:
            self._nearest = distance_transform_edt(~self.trav, return_distances=False, return_indices=True)
        return int(self._nearest[0][i, j]), int(self._nearest[1][i, j])

    def _route(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """-> (path cells to the goal cell, the grid it runs on and its shortcuts must stay on, whether that
        grid was the observed one). Observed cells first when `observed_first`, the start and goal cells
        always open; an empty path when the goal cell is unreachable."""
        if self.trav_observed is not None:
            grid = self.trav_observed
            saved = grid[start], grid[goal]
            grid[start] = grid[goal] = True
            path = astar(grid, start[0], start[1], goal[0], goal[1])
            if len(path):
                return path, grid, saved
            grid[start], grid[goal] = saved
        return astar(self.trav, start[0], start[1], goal[0], goal[1]), self.trav, None

    def target(self, x: float, z: float, gx: float, gz: float) -> Tuple[float, float]:
        """World (x, z) to steer at from (x, z) toward the goal (gx, gz)."""
        t0 = time.perf_counter()
        ax, az = self._cells(x, z)
        cx, cz = self._cells(gx, gz)
        start = self._start_cell(ax, az)
        if start is None:
            return gx, gz
        gi, gj = int(math.floor(cx)), int(math.floor(cz))
        nx, nz = self.trav.shape
        path, grid, saved = np.zeros((0, 2), np.int64), self.trav, None
        if 0 <= gi < nx and 0 <= gj < nz and self.trav[gi, gj]:
            path, grid, saved = self._route(start, (gi, gj))
        reached = len(path) > 0
        if not reached:
            ti, tj = nearest_reachable(self.trav, start[0], start[1], cx, cz)
            path = astar(self.trav, start[0], start[1], ti, tj)
            self.n_fallback += 1
        if grid is self.trav and self.trav_observed is not None:
            self.n_unobserved_plans += 1
        own = start == (int(math.floor(ax)), int(math.floor(az)))
        centres = path[1 if own else 0:].astype(np.float64) + 0.5
        if reached and len(centres):
            centres = centres[:-1]                          # the goal replaces its cell's centre
        vertices = np.concatenate([centres, np.array([[cx, cz]])])
        k = farthest_visible(grid, ax, az, vertices, self.episode_cfg.expert_lookahead_m / self.ground.res)
        if saved is not None:
            grid[start], grid[(gi, gj)] = saved
        if self.first_plan is None:
            self.first_plan = self._world(np.concatenate([[[ax, az]], vertices]))
        self.n_plans += 1
        self.plan_seconds += time.perf_counter() - t0
        return tuple(self._world(vertices[k]))

    def _decide(self, x: float, z: float, yaw: float, gx: float, gz: float) -> int:
        if math.hypot(gx - x, gz - z) <= self.episode_cfg.expert_stop_radius:
            return Action.STOP
        tx, tz = self.target(x, z, gx, gz)
        error = wrap_angle(heading_of(tx - x, tz - z) - yaw)
        if abs(error) > math.radians(self.episode_cfg.turn_deg) / 2.0:
            return Action.TURN_LEFT if error > 0 else Action.TURN_RIGHT
        return Action.FORWARD

    def _believed(self, pose) -> Tuple[float, float, float]:
        if not any(self.query_noise):
            return tuple(float(v) for v in pose)
        believed = np.asarray(pose, np.float32).reshape(1, 3).copy()
        jitter_(believed, *self.query_noise, self.rng)
        return tuple(float(v) for v in believed[0])

    def _chunk(self, pose: Tuple[float, float, float], goal_position) -> List[int]:
        """`exec_horizon` actions from `pose`, each planned at the pose the previous ones lead to on the
        discrete kinematics; ends at a STOP."""
        x, z, yaw = pose
        gx, gz = (float(v) for v in goal_position)
        turn = math.radians(self.episode_cfg.turn_deg)
        actions = []
        for _ in range(self.exec_horizon):
            action = self._decide(x, z, yaw, gx, gz)
            actions.append(action)
            if action == Action.STOP:
                break
            if action == Action.FORWARD:
                x, z = x - math.sin(yaw) * FORWARD_STEP_M, z - math.cos(yaw) * FORWARD_STEP_M
            else:
                yaw = float(wrap_angle(yaw + (turn if action == Action.TURN_LEFT else -turn)))
        return actions

    def __call__(self, pose, goal_position, stepper=None) -> int:
        x, z, yaw = (float(v) for v in pose)
        if self._last is not None:
            lx, lz, last_action = self._last
            if last_action == Action.FORWARD and math.hypot(x - lx, z - lz) < 0.5 * FORWARD_STEP_M:
                self.n_blocked += 1
                self.last_blocked = (x, z, yaw)
                if self.cfg.bump_marking:
                    self._mark_bump(x, z, yaw)
        if not self.queue:
            self.queue.extend(self._chunk(self._believed(pose), goal_position))
        action = self.queue.pop(0)
        self._last = (x, z, int(action))
        return action


# ---------------------------------------------------------------- evaluation
def tour_frames(store: TourStore, traj_root: str, tour: str):
    """The context the learned planners read for this tour, through their own store -> (meta, frame
    indices, depth (K, H, W) float32, T_world_camera (K, 4, 4) float64 at the store's evaluation poses,
    the navmesh raster at the tour's floor for diagnostics)."""
    meta, frames = store.meta(tour), store.keyframes(tour)
    _, pose = store.eval_context(tour)                     # (K, 3) x, z, yaw under the store's keyframe noise
    with h5py.File(os.path.join(traj_root, tour), "r") as f:
        depth = f["depth"][frames].astype(np.float32)
        recorded = f["camera_pose_world"][frames].astype(np.float64)
        navigable = f["scene_topdown"][:].astype(bool)
    camera_height = recorded[:, 1, 3]
    assert np.allclose(camera_to_world(meta["pose"][frames].astype(np.float64), camera_height), recorded, atol=1e-4), \
        f"{tour}: recorded cameras are not the store's poses"
    if not (store.cfg.kf_noise_pos or store.cfg.kf_noise_yaw):
        return meta, frames, depth, recorded, navigable
    return meta, frames, depth, camera_to_world(pose.astype(np.float64), camera_height), navigable


AGREEMENT_MARGIN_M = 0.5            # the diagnostic region reaches this far past the observed cells, to take in walls


def map_agreement(ground: GroundMap, trav: np.ndarray, meta: dict, navigable: np.ndarray) -> dict:
    """Diagnostics only (the policy never sees the navmesh): the 2-D map against the navmesh raster at
    the tour's floor, over the cells the tour observed (coverage 2, all navigable) and the cells within
    AGREEMENT_MARGIN_M of them (walls and furniture included)."""
    region = binary_dilation(meta["coverage"] == 2, structure=footprint(AGREEMENT_MARGIN_M, meta["mpp"]))
    rows, cols = np.nonzero(region)
    x = meta["origin_position"][0] + (cols + 0.5) * meta["mpp"]
    z = meta["origin_position"][1] + (rows + 0.5) * meta["mpp"]
    i = np.floor(x / ground.res).astype(np.int64) - ground.key0[0]
    j = np.floor(z / ground.res).astype(np.int64) - ground.key0[1]
    inside = (i >= 0) & (i < trav.shape[0]) & (j >= 0) & (j < trav.shape[1])
    nav = navigable[rows, cols]
    ii, jj = np.clip(i, 0, trav.shape[0] - 1), np.clip(j, 0, trav.shape[1] - 1)
    t = trav[ii, jj] & inside
    obstacle = ground.obstacle[ii, jj] & inside
    floor = ground.floor[ii, jj] & inside
    observed = ground.observed[ii, jj] & inside

    def frac(num, den):
        return float((num & den).sum() / max(den.sum(), 1))

    return dict(cells=int(len(rows)), navigable=float(nav.mean()) if len(nav) else 0.0,
                observed=float(observed.mean()) if len(nav) else 0.0,
                trav_given_nav=frac(t, nav), nav_given_trav=frac(nav, t),
                obstacle_given_nav=frac(obstacle, nav), floor_given_nav=frac(floor, nav),
                trav_given_not_nav=frac(t, ~nav), trav_observed_given_not_nav=frac(t & observed, ~nav))


_JOB: dict = {}                     # set before the worker pool forks


def run_tour(tour_rows: Tuple[str, List[int]]) -> Tuple[List[dict], dict]:
    tour, rows = tour_rows
    cfg, episode_cfg, bank = _JOB["cfg"], _JOB["episode_cfg"], _JOB["bank"]
    t0 = time.perf_counter()
    meta, frames, depth, T_world_cameras, navigable = tour_frames(_JOB["store"], _JOB["traj_root"], tour)
    t1 = time.perf_counter()
    grid, floor_ref = build_map(depth, T_world_cameras, cfg)
    ground = squash(grid, floor_ref, cfg.band_top)
    trav = traversable(ground, cfg)
    t2 = time.perf_counter()
    stats = dict(tour=tour, frames=int(len(frames)), grid_shape=list(grid.logodds.shape),
                 known_voxels=int(grid.known.sum()), occupied_voxels=int((grid.known & (grid.logodds >= 0)).sum()),
                 read_seconds=round(t1 - t0, 3), map_seconds=round(t2 - t1, 3),
                 agreement=map_agreement(ground, trav, meta, navigable))
    del grid
    stepper = Stepper(episode_cfg).use_pathfinder(load_pathfinder(meta["scene_glb"]))
    results = []
    for i in rows:
        policy = OccupancyAStarPolicy(ground, trav, cfg, episode_cfg, _JOB["exec_horizon"], _JOB["query_noise"],
                                      np.random.default_rng(int(i)))
        rollout = policy_rollout(stepper, policy, bank.episode(i), episode_cfg, record_trajectory=True)
        trajectory = np.asarray(rollout["traj"], np.float32)
        path_len = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum()) if len(trajectory) > 1 else 0.0
        failure = "" if rollout["success"] else ("wrong_stop" if rollout["stopped"] else "never_stopped")
        blocked = cell_state(ground, *cell_ahead(ground, *policy.last_blocked)) if policy.last_blocked else ""
        result = row_result(bank, i, rollout, path_len=path_len, n_added=0, n_rendered=0, failure=failure,
                            n_blocked=policy.n_blocked, n_bumps=policy.n_bumps, last_blocked_map=blocked,
                            last_blocked=[round(v, 5) for v in policy.last_blocked] if policy.last_blocked else None,
                            n_fallback=policy.n_fallback, n_plans=policy.n_plans,
                            n_unobserved_plans=policy.n_unobserved_plans,
                            plan_ms=round(1e3 * policy.plan_seconds / max(policy.n_plans, 1), 3),
                            first_plan=policy.first_plan.round(3).tolist() if policy.first_plan is not None else None)
        if not _JOB["save_traj"]:
            result["traj"] = result["first_plan"] = None
        results.append(result)
    stats["episode_seconds"] = round(time.perf_counter() - t2, 3)
    return results, stats


def _warm_up() -> None:
    """Compile the kernels once in the parent, so forked workers inherit them."""
    trav = np.ones((4, 4), np.bool_)
    grid = np.zeros((3, 3, 3), np.float32)
    insert_scan(grid, np.zeros((3, 3, 3), np.bool_), np.zeros((3, 3, 3), np.int32), 0, np.full(3, 0.01),
                np.full((1, 3), 0.12), np.zeros(3, np.int64), 0.05, -1.0, 0.85, -0.4, -2.0, 3.5)
    astar(trav, 0, 0, 3, 3)
    nearest_reachable(trav, 0, 0, 3.5, 3.5)
    farthest_visible(trav, 0.5, 0.5, np.array([[1.5, 1.5], [3.5, 3.5]]), 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--token-root", required=True, help="token stores: context frame selection and tour metadata")
    p.add_argument("--traj-root", required=True, help="raw exploration shards: depth and camera_pose_world")
    p.add_argument("--bank", required=True)
    p.add_argument("--eta", type=int, default=10, help="context frames: coverage-gate eta; 0 = every frame")
    p.add_argument("--max-kf", type=int, default=768, help="context frames beyond this are uniformly subsampled")
    p.add_argument("--exec-horizon", type=int, default=1, help="actions committed per pose read")
    p.add_argument("--kf-noise-pos", type=float, default=0.0,
                   help="metres, uniform jitter on each context frame's position (the store's fixed draw)")
    p.add_argument("--kf-noise-yaw", type=float, default=0.0, help="radians, uniform jitter on each context frame's yaw")
    p.add_argument("--query-noise-pos", type=float, default=0.0,
                   help="metres, uniform error on the agent's position at every query")
    p.add_argument("--query-noise-yaw", type=float, default=0.0, help="radians, uniform error on the agent's yaw")
    p.add_argument("--rows", default="", help="comma-separated bank rows (default: all)")
    p.add_argument("--limit", type=int, default=0, help="evaluate the first N rows only")
    p.add_argument("--workers", type=int, default=8, help="tours in parallel, one process each")
    p.add_argument("--save-traj", action="store_true", help="store each row's trajectory and first plan")
    p.add_argument("--out", required=True)
    add_config_args(p, ClassicalConfig)
    args = p.parse_args()
    cfg = config_from_args(ClassicalConfig, args)

    bank = EvalBank(args.bank)
    rows = (np.array(sorted(int(x) for x in args.rows.split(","))) if args.rows
            else np.arange(min(args.limit, len(bank)) if args.limit else len(bank)))
    by_tour: dict = {}
    for i in rows:
        by_tour.setdefault(bank.tours[i], []).append(int(i))
    jobs = sorted(by_tour.items(), key=lambda kv: -len(kv[1]))
    store = TourStore(DataConfig(token_root=args.token_root, eta=args.eta, max_kf=args.max_kf,
                                 kf_noise_pos=args.kf_noise_pos, kf_noise_yaw=args.kf_noise_yaw, zero_tokens=True))
    query_noise = (args.query_noise_pos, args.query_noise_yaw)
    _JOB.update(cfg=cfg, episode_cfg=EpisodeConfig(), bank=bank, store=store, traj_root=args.traj_root,
                exec_horizon=args.exec_horizon, query_noise=query_noise, save_traj=args.save_traj)
    _warm_up()
    t0 = time.perf_counter()
    results, tours = [], []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for tour_results, stats in pool.imap_unordered(run_tour, jobs):
            results += tour_results
            tours.append(stats)
            if len(tours) % 25 == 0 or len(tours) == len(jobs):
                print(f"[{len(tours)}/{len(jobs)} tours, {len(results)} rows, {time.perf_counter() - t0:.0f} s] "
                      f"running success {np.mean([r['success'] for r in results]):.3f}", flush=True)
    results.sort(key=lambda r: r["row"])
    out = dict(**summarize_results(results), method="octomap_astar", config=dataclasses.asdict(cfg),
               eta=args.eta, max_kf=args.max_kf, exec_horizon=args.exec_horizon,
               kf_noise=[args.kf_noise_pos, args.kf_noise_yaw], query_noise=list(query_noise), accum=False,
               bank=args.bank, token_root=args.token_root, traj_root=args.traj_root,
               wall_seconds=round(time.perf_counter() - t0, 1), tours=tours, results=results)
    write_json(out, args.out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("results", "tours")}))


if __name__ == "__main__":
    main()
