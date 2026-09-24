#!/usr/bin/env python3
# =================================================================================================
# apcbf_core7.py  —  Dual-kinematic CBF: holonomic 3D QP (vx, vy, ω) or unicycle 2D QP (v, ω).
#
# WHAT'S NEW vs apcbf_core6.py:
#   - CBFParams gains kinematic_model: 'holonomic' or 'unicycle'.
#   - Each multi-obstacle filter checks the model and builds the appropriate QP:
#       holonomic: 3 control inputs (vx, vy, ω)  → QP variables [vx, vy, ω, ...]
#       unicycle:  2 control inputs (v, ω)        → QP variables [v, ω, ...]
#     The unicycle drops gy from the constraint because the robot cannot move laterally.
#     CBFResult.vy = 0.0 always in unicycle mode.
#   - The barrier computation (compute_barrier) is unchanged — it returns the full
#     5-tuple (rho, h, gx, gy, gw) regardless of model. The filter selects which
#     gradient components to use.
#
# DEPENDENCIES: numpy, quadprog  (pip install quadprog --break-system-packages)
# =================================================================================================


import math
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import quadprog


# -------------------------------------------------------------------------------------------------
# Configuration bundle
# -------------------------------------------------------------------------------------------------
@dataclass
class CBFParams:
    # ---- Core CBF ----
    alpha: float = 1.0             # class-K gain: grad_h·u >= -α·h
    d_safe: float = 0.4           # safety radius for point-distance mode (meters)
    # ---- Shared slack (distance_multi_slack) ----
    p_slack: float = 4.0          # quadratic penalty on shared slack ε
    # ---- Weighted slack (distance_multi_slack_weighted) ----
    p_base: float = 5.0           # base weight: penalty_i = p_base / (h_i + eta_w)²
    eta_w: float = 0.05           # regularizer preventing div-by-zero as h→0
    # ---- BP-SDF shape-aware barrier ----
    barrier_mode: str = 'point'   # 'point' = circle, 'bpsdf' = shape-aware
    sdf_margin: float = 0.01      # clearance beyond fitted shape (meters)
    sdf_config_path: str = ''     # path to bp_sdf_coeffs.json
    # ---- Angular velocity cost ----
    w_weight: float = 1.0         # QP cost on ω deviation. >1 = prefer clipping translation
    # ---- Kinematic model ----
    kinematic_model: str = 'holonomic'  # 'holonomic' = 3D (vx,vy,ω), 'unicycle' = 2D (v,ω)


# -------------------------------------------------------------------------------------------------
# BP-SDF runtime evaluator — loaded once, called per obstacle point per filter cycle.
# -------------------------------------------------------------------------------------------------
class BPSDFEvaluator:
    """Loads fitted Bernstein polynomial SDF coefficients and evaluates h + grad at query points."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        self.degree = int(cfg['degree'])
        self.w = np.array(cfg['coefficients'], dtype=np.float64)
        self.domain = (tuple(cfg['domain']['x']), tuple(cfg['domain']['y']))
        self.robot_width = cfg['robot_width']
        self.robot_length = cfg['robot_length']
        d1 = self.degree + 1
        assert len(self.w) == d1 * d1, f"Expected {d1*d1} coeffs, got {len(self.w)}"

    def evaluate(self, px: float, py: float) -> Tuple[float, float, float]:
        """
        Evaluate BP-SDF at a single point (px, py) in the robot body frame.

        Returns:
            value: SDF value (positive = outside robot, negative = inside)
            grad_x, grad_y: spatial gradient of the SDF
        """
        px_a = np.array([px], dtype=np.float64)
        py_a = np.array([py], dtype=np.float64)

        (xmin, xmax), (ymin, ymax) = self.domain
        tx = np.clip((px_a - xmin) / (xmax - xmin), 0, 1)
        ty = np.clip((py_a - ymin) / (ymax - ymin), 0, 1)

        d = self.degree
        d1 = d + 1

        # --- 1D Bernstein basis and derivatives ---
        Bx = self._bernstein_1d(tx, d)           # (1, d1)
        By = self._bernstein_1d(ty, d)
        dBx = self._bernstein_1d_deriv(tx, d, Bx_lower=self._bernstein_1d(tx, d - 1))
        dBy = self._bernstein_1d_deriv(ty, d, Bx_lower=self._bernstein_1d(ty, d - 1))

        dtx_dpx = 1.0 / (xmax - xmin)
        dty_dpy = 1.0 / (ymax - ymin)

        # --- Tensor product + gradient in one pass ---
        value = 0.0
        grad_x = 0.0
        grad_y = 0.0
        for i in range(d1):
            for j in range(d1):
                wij = self.w[i * d1 + j]
                bx_i = Bx[0, i]
                by_j = By[0, j]
                value += wij * bx_i * by_j
                grad_x += wij * dBx[0, i] * by_j * dtx_dpx
                grad_y += wij * bx_i * dBy[0, j] * dty_dpy

        return float(value), float(grad_x), float(grad_y)

    @staticmethod
    def _bernstein_1d(t, degree):
        """Evaluate all Bernstein basis polynomials at parameter t. Shape (1, degree+1)."""
        n = degree
        k = np.arange(n + 1)
        from math import comb
        binom = np.array([comb(n, ki) for ki in k], dtype=np.float64)
        T = t[:, None]
        return binom[None, :] * (T ** k[None, :]) * ((1 - T) ** (n - k)[None, :])

    @staticmethod
    def _bernstein_1d_deriv(t, degree, Bx_lower):
        """Derivative of Bernstein basis. Uses pre-computed degree-(n-1) basis."""
        n = degree
        deriv = np.zeros((len(t), n + 1))
        for k in range(n + 1):
            if k > 0:
                deriv[:, k] += n * Bx_lower[:, k - 1]
            if k < n:
                deriv[:, k] -= n * Bx_lower[:, k]
        return deriv


# -------------------------------------------------------------------------------------------------
# Singleton SDF loader (loads once per process, reused across all filter calls).
# -------------------------------------------------------------------------------------------------
_sdf_cache: dict = {}

def _get_sdf_evaluator(config_path: str) -> BPSDFEvaluator:
    if config_path not in _sdf_cache:
        _sdf_cache[config_path] = BPSDFEvaluator(config_path)
    return _sdf_cache[config_path]


# -------------------------------------------------------------------------------------------------
# compute_barrier():  THE SINGLE INTEGRATION POINT.
#
#   Given an obstacle point (cx, cy) in base_link frame, returns (rho, h, gx, gy, gw).
#
#   'point' mode (original):  rho = ||p||,  h = rho - d_safe,  grad = -p/||p||, gw = 0
#   'bpsdf' mode (new):       sdf = Ψ(p)ᵀw,  h = sdf - margin,  grad = ∇Ψ(p)ᵀw
#       rho is still Euclidean distance (for logging/haptics), but h and grad encode the shape.
#
#   The angular gradient gw comes from how a static obstacle moves in body frame under rotation:
#       ṗ_body = [-vx + ω·cy, -vy - ω·cx]
#   so ḣ = gx·vx + gy·vy + gw·ω  where  gw = -gx·cy + gy·cx.
#   For point-distance, gw = 0 identically (rotation doesn't change point-to-point distance).
# -------------------------------------------------------------------------------------------------
def compute_barrier(cx: float, cy: float, p: CBFParams) -> Tuple[float, float, float, float, float]:
    """
    Returns (rho, h, gx, gy, gw) for one obstacle point.
    
    rho:  Euclidean distance from robot center to obstacle (always, for logging/haptics)
    h:    barrier function value (positive = safe)
    gx,gy: gradient of h w.r.t. robot translational velocity
    gw:   gradient of h w.r.t. robot angular velocity
    """
    rho = math.hypot(cx, cy)
    if rho < 1e-9:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    if p.barrier_mode == 'bpsdf':
        if not p.sdf_config_path:
            raise ValueError("barrier_mode='bpsdf' but sdf_config_path is empty")
        sdf = _get_sdf_evaluator(p.sdf_config_path)
        sdf_val, sdf_gx, sdf_gy = sdf.evaluate(cx, cy)
        h = sdf_val - p.sdf_margin
        gx = -sdf_gx
        gy = -sdf_gy
        # Angular gradient:  gw = sdf_gx*cy - sdf_gy*cx = -gx*cy + gy*cx
        gw = -gx * cy + gy * cx
    else:
        # Original point-distance model
        h = rho - p.d_safe
        gx = -cx / rho
        gy = -cy / rho
        # gw = -gx*cy + gy*cx = (cx/rho)*cy - (cy/rho)*cx = 0
        gw = 0.0

    return (rho, h, gx, gy, gw)


# -------------------------------------------------------------------------------------------------
# Diagnostics. Two new optional fields (n_obstacles, n_active) for the multi-obstacle QP; they
# default to None so the single-obstacle filters that don't set them still construct fine.
# -------------------------------------------------------------------------------------------------
@dataclass
class CBFResult:
    vx: float
    vy: float
    wz: float                              # filtered angular velocity (NEW in core6)
    intervened: bool
    rho: float | None
    h: float | None
    psi: float | None
    constraint_lhs: float | None
    constraint_ok: bool | None
    vsafe_align: float | None
    grad_h_norm: float | None
    n_obstacles: int | None = None
    n_active: int | None = None
    eps: float | None = None


# -------------------------------------------------------------------------------------------------
# _build_qp_components():  Shared helper for all multi-obstacle filters.
#
#   Iterates obstacle points, calls compute_barrier, and builds QP arrays
#   for either holonomic (3D) or unicycle (2D) kinematics.
#
#   Returns None on degenerate obstacle (rho=0), otherwise returns:
#       (diag, grad_cols, b_list, n_ctrl)
#   where n_ctrl is 2 (unicycle) or 3 (holonomic).
# -------------------------------------------------------------------------------------------------
def _build_qp_components(obstacle_points, p: CBFParams):
    """
    Returns (diag, grad_cols, b_list, n_ctrl) or None if degenerate.
    
    diag:      list of (rho, h, gx, gy, gw) per obstacle
    grad_cols: list of gradient vectors (length n_ctrl) per obstacle
    b_list:    list of -alpha*h per obstacle
    n_ctrl:    2 for unicycle, 3 for holonomic
    """
    unicycle = (p.kinematic_model == 'unicycle')
    n_ctrl = 2 if unicycle else 3

    diag = []
    grad_cols = []
    b_list = []

    for (cx, cy) in obstacle_points:
        rho, h, gx, gy, gw = compute_barrier(cx, cy, p)
        if rho < 1e-9:
            return None
        diag.append((rho, h, gx, gy, gw))
        if unicycle:
            grad_cols.append([gx, gw])            # 2D: [v, ω]  — drop gy
        else:
            grad_cols.append([gx, gy, gw])         # 3D: [vx, vy, ω]
        b_list.append(-p.alpha * h)

    return (diag, grad_cols, b_list, n_ctrl)


def _make_vdes(vdes_x, vdes_y, vdes_wz, p: CBFParams):
    """Build the desired velocity vector for the appropriate kinematic model."""
    if p.kinematic_model == 'unicycle':
        return np.array([float(vdes_x), float(vdes_wz)])    # [v, ω]
    else:
        return np.array([float(vdes_x), float(vdes_y), float(vdes_wz)])  # [vx, vy, ω]


def _make_cost(vdes, ww, p: CBFParams):
    """Build QP cost matrix G and linear term a for the velocity variables."""
    if p.kinematic_model == 'unicycle':
        G = np.diag([2.0, 2.0 * ww])
        a = np.array([2.0 * vdes[0], 2.0 * ww * vdes[1]])
    else:
        G = np.diag([2.0, 2.0, 2.0 * ww])
        a = np.array([2.0 * vdes[0], 2.0 * vdes[1], 2.0 * ww * vdes[2]])
    return G, a


def _extract_result(sol_v, vdes_x, vdes_y, vdes_wz, p: CBFParams):
    """Extract (vfx, vfy, wfz, intervened) from QP solution vector."""
    if p.kinematic_model == 'unicycle':
        vfx = float(sol_v[0])
        vfy = 0.0                                 # unicycle cannot move laterally
        wfz = float(sol_v[1])
        intervened = not (abs(vfx - vdes_x) < 1e-9 and abs(wfz - vdes_wz) < 1e-9)
    else:
        vfx = float(sol_v[0])
        vfy = float(sol_v[1])
        wfz = float(sol_v[2])
        intervened = not (abs(vfx - vdes_x) < 1e-9 and abs(vfy - vdes_y) < 1e-9
                          and abs(wfz - vdes_wz) < 1e-9)
    return vfx, vfy, wfz, intervened


def _check_constraints(diag, vfx, vfy, wfz, alpha, constraint_tol, p: CBFParams):
    """Check constraint satisfaction and count active constraints."""
    n_active = 0
    constraint_ok = True
    for (rho, h, gx, gy, gw) in diag:
        if p.kinematic_model == 'unicycle':
            dot = gx * vfx + gw * wfz
        else:
            dot = gx * vfx + gy * vfy + gw * wfz
        if dot + alpha * h < -constraint_tol:
            constraint_ok = False
        if abs(dot - (-alpha * h)) < 1e-4:
            n_active += 1
    return constraint_ok, n_active


# -------------------------------------------------------------------------------------------------
# Multi-obstacle hard QP — dual kinematic: holonomic 3D or unicycle 2D.
# -------------------------------------------------------------------------------------------------
def cbf_filter_multi(vdes_x, vdes_y, obstacle_points, p: CBFParams,
                     vdes_wz: float = 0.0, constraint_tol: float = 1e-6) -> CBFResult:
    """Hard QP with no slack. Holonomic solves over (vx,vy,ω); unicycle over (v,ω)."""
    if not obstacle_points:
        vy_out = float(vdes_y) if p.kinematic_model != 'unicycle' else 0.0
        return CBFResult(vdes_x, vy_out, vdes_wz, False, None, None, None, None, None, None, None,
                         n_obstacles=0, n_active=0)

    qp = _build_qp_components(obstacle_points, p)
    if qp is None:
        return CBFResult(0.0, 0.0, 0.0, True, 0.0, None, None, None, None, None, None,
                         n_obstacles=len(obstacle_points), n_active=None)
    diag, grad_cols, b_list, n_ctrl = qp

    ww = p.w_weight if p.w_weight > 0 else 1.0
    vdes = _make_vdes(vdes_x, vdes_y, vdes_wz, p)
    G, a = _make_cost(vdes, ww, p)
    C = np.array(grad_cols).T
    b = np.array(b_list)

    try:
        sol = quadprog.solve_qp(G, a, C, b, 0)
        v = sol[0]
    except ValueError:
        return CBFResult(0.0, 0.0, 0.0, True, None, None, None, None, False, None, None,
                         n_obstacles=len(diag), n_active=None)

    vfx, vfy, wfz, intervened = _extract_result(v, vdes_x, vdes_y, vdes_wz, p)
    constraint_ok, n_active = _check_constraints(diag, vfx, vfy, wfz, p.alpha, constraint_tol, p)

    return CBFResult(vfx, vfy, wfz, intervened,
                     min(d[0] for d in diag), min(d[1] for d in diag), None,
                     None, constraint_ok, None, 1.0,
                     n_obstacles=len(diag), n_active=n_active)


# -------------------------------------------------------------------------------------------------
# Multi-obstacle shared-slack QP — dual kinematic.
# -------------------------------------------------------------------------------------------------
def cbf_filter_multi_slack(vdes_x, vdes_y, obstacle_points, p: CBFParams,
                           vdes_wz: float = 0.0, constraint_tol: float = 1e-6) -> CBFResult:
    """Shared slack ε. Holonomic: (vx,vy,ω,ε); unicycle: (v,ω,ε)."""
    if not obstacle_points:
        vy_out = float(vdes_y) if p.kinematic_model != 'unicycle' else 0.0
        return CBFResult(vdes_x, vy_out, vdes_wz, False, None, None, None, None, None, None, None,
                         n_obstacles=0, n_active=0, eps=0.0)

    qp = _build_qp_components(obstacle_points, p)
    if qp is None:
        return CBFResult(0.0, 0.0, 0.0, True, 0.0, None, None, None, None, None, None,
                         n_obstacles=len(obstacle_points), n_active=None, eps=None)
    diag, grad_cols, b_list_raw, n_ctrl = qp

    ww = p.w_weight if p.w_weight > 0 else 1.0
    vdes = _make_vdes(vdes_x, vdes_y, vdes_wz, p)
    G_vel, a_vel = _make_cost(vdes, ww, p)

    # Extend with slack variable: x = [v..., eps]
    nvars = n_ctrl + 1
    G = np.zeros((nvars, nvars))
    G[:n_ctrl, :n_ctrl] = G_vel
    G[n_ctrl, n_ctrl] = 2.0 * p.p_slack
    a = np.zeros(nvars)
    a[:n_ctrl] = a_vel
    # a[n_ctrl] = 0.0  (no linear slack cost for quadratic penalty)

    # Constraints: eps >= 0, then [grad_i, 1] . x >= -alpha*h_i
    cols = [np.zeros(nvars)]
    cols[0][n_ctrl] = 1.0         # eps >= 0
    b_list = [0.0]
    for i, gc in enumerate(grad_cols):
        col = np.zeros(nvars)
        col[:n_ctrl] = gc
        col[n_ctrl] = 1.0        # +eps softens this constraint
        cols.append(col)
        b_list.append(b_list_raw[i])

    C = np.array(cols).T
    b = np.array(b_list)

    try:
        sol = quadprog.solve_qp(G, a, C, b, 0)
        x = sol[0]
    except ValueError:
        return CBFResult(0.0, 0.0, 0.0, True, None, None, None, None, False, None, None,
                         n_obstacles=len(diag), n_active=None, eps=None)

    vfx, vfy, wfz, intervened = _extract_result(x[:n_ctrl], vdes_x, vdes_y, vdes_wz, p)
    eps = max(0.0, float(x[n_ctrl]))

    constraint_ok, n_active = _check_constraints(diag, vfx, vfy, wfz, p.alpha, constraint_tol, p)

    return CBFResult(vfx, vfy, wfz, intervened,
                     min(d[0] for d in diag), min(d[1] for d in diag), None,
                     None, constraint_ok, None, 1.0,
                     n_obstacles=len(diag), n_active=n_active, eps=eps)


# -------------------------------------------------------------------------------------------------
# Multi-obstacle per-obstacle-weighted slack QP — dual kinematic.
# -------------------------------------------------------------------------------------------------
def cbf_filter_multi_slack_weighted(vdes_x, vdes_y, obstacle_points, p: CBFParams,
                                    vdes_wz: float = 0.0, constraint_tol: float = 1e-6) -> CBFResult:
    """Per-obstacle slack εᵢ, weighted by proximity. Holonomic or unicycle."""
    if not obstacle_points:
        vy_out = float(vdes_y) if p.kinematic_model != 'unicycle' else 0.0
        return CBFResult(vdes_x, vy_out, vdes_wz, False, None, None, None, None, None, None, None,
                         n_obstacles=0, n_active=0, eps=0.0)

    qp = _build_qp_components(obstacle_points, p)
    if qp is None:
        return CBFResult(0.0, 0.0, 0.0, True, 0.0, None, None, None, None, None, None,
                         n_obstacles=len(obstacle_points), n_active=None, eps=None)
    diag, grad_cols, b_list_raw, n_ctrl = qp
    N = len(diag)

    ww = p.w_weight if p.w_weight > 0 else 1.0
    vdes = _make_vdes(vdes_x, vdes_y, vdes_wz, p)
    G_vel, a_vel = _make_cost(vdes, ww, p)

    # Variables: [v...(n_ctrl), eps_1, ..., eps_N]
    nvars = n_ctrl + N
    Gdiag = np.zeros(nvars)
    Gdiag[:n_ctrl] = np.diag(G_vel)
    for i, (rho, h, gx, gy, gw) in enumerate(diag):
        p_i = p.p_base / (h + p.eta_w) ** 2
        Gdiag[n_ctrl + i] = 2.0 * p_i
    G = np.diag(Gdiag)

    a = np.zeros(nvars)
    a[:n_ctrl] = a_vel

    # Constraints
    cols, b_list = [], []
    # (a) softened barriers: grad_i . v + eps_i >= -alpha*h_i
    for i, gc in enumerate(grad_cols):
        col = np.zeros(nvars)
        col[:n_ctrl] = gc
        col[n_ctrl + i] = 1.0
        cols.append(col)
        b_list.append(b_list_raw[i])
    # (b) eps_i >= 0
    for i in range(N):
        col = np.zeros(nvars)
        col[n_ctrl + i] = 1.0
        cols.append(col)
        b_list.append(0.0)

    C = np.array(cols).T
    b = np.array(b_list)

    try:
        sol = quadprog.solve_qp(G, a, C, b, 0)
        x = sol[0]
    except ValueError:
        return CBFResult(0.0, 0.0, 0.0, True, None, None, None, None, False, None, None,
                         n_obstacles=N, n_active=None, eps=None)

    vfx, vfy, wfz, intervened = _extract_result(x[:n_ctrl], vdes_x, vdes_y, vdes_wz, p)
    eps_vec = [max(0.0, float(x[n_ctrl + i])) for i in range(N)]
    eps_max = max(eps_vec) if eps_vec else 0.0

    constraint_ok, n_active = _check_constraints(diag, vfx, vfy, wfz, p.alpha, constraint_tol, p)

    return CBFResult(vfx, vfy, wfz, intervened,
                     min(d[0] for d in diag), min(d[1] for d in diag), None,
                     None, constraint_ok, None, 1.0,
                     n_obstacles=N, n_active=n_active, eps=eps_max)


# -------------------------------------------------------------------------------------------------
# One-line log string. Handles single-obstacle and multi-obstacle results.
# -------------------------------------------------------------------------------------------------
def format_result(r: CBFResult) -> str:
    rho_s = f'{r.rho:.3f}' if r.rho is not None else '  n/a'
    h_s   = f'{r.h:+.4f}'  if r.h   is not None else '   n/a'
    psi_s = f'{r.psi:+.4f}' if r.psi is not None else '   n/a'

    if r.n_obstacles is not None:
        act = f'{r.n_active}' if r.n_active is not None else 'STOP'
        eps_s = f' eps={r.eps:.4f}' if r.eps is not None else ''
        multi = f"  obstacles={r.n_obstacles} active={act}{eps_s}\n"
    else:
        multi = ""

    if r.constraint_lhs is not None:
        ok_s = 'OK ' if r.constraint_ok else 'FAIL'
        check = (f"  CHECK  : constraint={r.constraint_lhs:+.4f} [{ok_s}]   "
                 f"vsafe_align={r.vsafe_align:+.4f}   |grad_h|={r.grad_h_norm:.4f}\n")
    elif r.constraint_ok is not None:
        # multi case: no single constraint_lhs, but report feasibility
        check = f"  CHECK  : all-constraints-ok={r.constraint_ok}\n"
    else:
        check = ""

    return (f"\n  rho: {rho_s}   h(min): {h_s}   Psi: {psi_s}   intervened: {r.intervened}\n"
            f"{multi}{check}"
            f"  vfinal: ({r.vx:+.3f}, {r.vy:+.3f}, wz={r.wz:+.3f})")


# ============================ self-test ============================
if __name__ == '__main__':
    p = CBFParams(alpha=1.0, d_safe=0.4, p_slack=50.0, p_base=5.0, eta_w=0.05)

    def row(name, r):
        eps = f"{r.eps:.4f}" if r.eps is not None else "  n/a"
        print(f"    {name:18s} v=({r.vx:+.3f},{r.vy:+.3f}) wz={r.wz:+.3f}  eps={eps}  active={r.n_active}")

    scenarios = [
        ("wide gap",        0.3, 0.0, [(0.5,0.6),(0.5,-0.6)]),
        ("narrow gap",      0.3, 0.0, [(0.3,0.35),(0.3,-0.35)]),
        ("tight gap",       0.3, 0.0, [(0.3,0.25),(0.3,-0.25)]),
        ("boxed in",        0.3, 0.0, [(0.2,0.0),(-0.2,0.0),(0.0,0.2),(0.0,-0.2)]),
    ]
    for name, vx, vy, obs in scenarios:
        print(f"\n--- {name}  (obstacles={len(obs)}) ---")
        row("hard",     cbf_filter_multi(vx, vy, obs, p))
        row("slack",    cbf_filter_multi_slack(vx, vy, obs, p))
        row("weighted", cbf_filter_multi_slack_weighted(vx, vy, obs, p))

    # ---- Dual kinematic comparison ----
    print("\n" + "="*70)
    print("DUAL KINEMATIC: holonomic vs unicycle")
    print("="*70)
    obs = [(0.25, 0.1)]
    for model in ['holonomic', 'unicycle']:
        pp = CBFParams(alpha=1.0, d_safe=0.15, kinematic_model=model)
        r = cbf_filter_multi_slack_weighted(0.3, 0.1, obs, pp, vdes_wz=0.5)
        print(f"  [{model:10s}] v=({r.vx:+.3f},{r.vy:+.3f}) wz={r.wz:+.3f}  eps={r.eps:.4f}")

    # ---- BP-SDF test ----
    sdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bp_sdf_coeffs.json')
    if os.path.exists(sdf_path):
        print("\n" + "="*70)
        print("BP-SDF shape-aware barrier")
        print("="*70)
        p_bpsdf = CBFParams(alpha=1.0, sdf_margin=0.01, barrier_mode='bpsdf',
                            sdf_config_path=sdf_path)
        for label, obs in [("front", [(0.25, 0.0)]), ("corner", [(0.20, 0.15)]),
                           ("side", [(0.0, 0.20)])]:
            rho, h, gx, gy, gw = compute_barrier(obs[0][0], obs[0][1], p_bpsdf)
            r = cbf_filter_multi(0.3, 0.0, obs, p_bpsdf, vdes_wz=0.5)
            print(f"  {label:6s}: h={h:+.4f} gw={gw:+.4f} => "
                  f"v=({r.vx:+.3f},{r.vy:+.3f}) wz={r.wz:+.3f}")
    else:
        print(f"\n(skipping BP-SDF test — {sdf_path} not found)")