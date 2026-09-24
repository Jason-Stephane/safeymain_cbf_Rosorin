#!/usr/bin/env python3
# =================================================================================================
# apcbf_core2.py  —  Pure, ROS-free CBF math (single-obstacle closed-form + multi-obstacle QP)
#
# This is apcbf_core.py with ONE addition: cbf_filter_multi(), a distance-barrier CBF that enforces
# ALL obstacle constraints SIMULTANEOUSLY via a QP (quadprog), so the robot can pass *between*
# obstacles. The two original closed-form filters (cbf_filter, cbf_filter_distance) are unchanged.
#
# DEPENDENCIES: numpy, quadprog  (pip install quadprog --break-system-packages)
# =================================================================================================

import math
from dataclasses import dataclass

import numpy as np
import quadprog


# -------------------------------------------------------------------------------------------------
# Configuration bundle
# -------------------------------------------------------------------------------------------------
@dataclass
class CBFParams:
    delta: float = 0.001
    alpha: float = 1.0
    k_rep: float = 0.375
    rho_0: float = 1.5
    rho_cap: float = 0.5
    use_rho_cap: bool = False
    d_safe: float = 0.4
    p_slack: float = 4.0      # slack penalty for the soft-constrained multi QP. Larger => slack is
                               # more expensive => constraints stay harder (safer, may stop sooner).
                               # Smaller => relaxes more readily (gentler in tight gaps, less safe).
    p_slack_lin: float = 5.0   # LINEAR slack penalty (for cbf_filter_multi_slack_linear). Acts as a
                               # threshold: slack stays 0 until constraint pressure exceeds this.
    p_base: float = 5.0        # base weight for DISTANCE-WEIGHTED slack. Per-obstacle penalty is
                               # p_base / (h_i + eta_w)^2, so near obstacles are expensive to relax.
    eta_w: float = 0.05        # regularizer keeping the distance-weighted penalty finite at h->0.
    p_elastic_lin: float = 5.0 # ELASTIC mode: LINEAR slack weight  (the L1 part -> dead-zone).
    p_elastic_quad: float = 50.0  # ELASTIC mode: QUADRATIC slack weight (the L2 part -> smooth ease).


# -------------------------------------------------------------------------------------------------
# Diagnostics. Two new optional fields (n_obstacles, n_active) for the multi-obstacle QP; they
# default to None so the single-obstacle filters that don't set them still construct fine.
# -------------------------------------------------------------------------------------------------
@dataclass
class CBFResult:
    vx: float
    vy: float
    intervened: bool
    rho: float | None
    h: float | None
    psi: float | None
    constraint_lhs: float | None
    constraint_ok: bool | None
    vsafe_align: float | None
    grad_h_norm: float | None
    n_obstacles: int | None = None     # how many obstacle constraints the QP built
    n_active: int | None = None        # how many were binding at the solution
    eps: float | None = None           # slack used by the soft-constrained QP (0 => hard-satisfied)


# -------------------------------------------------------------------------------------------------
# Repulsive potential and gradient (used only by the APF-as-CBF filter)
# -------------------------------------------------------------------------------------------------
def compute_U_rep(rho: float, p: CBFParams) -> float:
    if rho > p.rho_0:
        return 0.0
    rho_eff = max(rho, p.rho_cap) if p.use_rho_cap else rho
    return 0.5 * p.k_rep * ((1.0 / rho_eff) - (1.0 / p.rho_0)) ** 2


def compute_grad_U_rep(cx: float, cy: float, rho: float, p: CBFParams):
    if rho > p.rho_0:
        return 0.0, 0.0
    rho_eff = max(rho, p.rho_cap) if p.use_rho_cap else rho
    u_x = -cx / rho
    u_y = -cy / rho
    force_mag = (p.k_rep / rho_eff ** 2) * ((1.0 / rho_eff) - (1.0 / p.rho_0))
    return -(u_x * force_mag), -(u_y * force_mag)


# -------------------------------------------------------------------------------------------------
# APF-as-CBF filter (Theorem 2) — single closest point, closed form. UNCHANGED.
# -------------------------------------------------------------------------------------------------
def cbf_filter(vdes_x, vdes_y, closest_xy, p: CBFParams, constraint_tol: float = 1e-6) -> CBFResult:
    if closest_xy is None:
        return CBFResult(vdes_x, vdes_y, False, None, None, None, None, None, None, None)
    cx, cy = closest_xy
    rho = math.hypot(cx, cy)
    if rho == 0.0:
        return CBFResult(0.0, 0.0, True, 0.0, None, None, None, None, None, None)

    U_rep = compute_U_rep(rho, p)
    h = 1.0 / (1.0 + U_rep) - p.delta
    gUx, gUy = compute_grad_U_rep(cx, cy, rho, p)
    denom = (1.0 + U_rep) ** 2
    grad_h_x = -gUx / denom
    grad_h_y = -gUy / denom

    psi = (grad_h_x * vdes_x + grad_h_y * vdes_y) + p.alpha * h
    if psi < 0.0:
        grad_h_sq = grad_h_x ** 2 + grad_h_y ** 2
        if grad_h_sq < 1e-12:
            v_safe_x = v_safe_y = 0.0
        else:
            scale = psi / grad_h_sq
            v_safe_x = -grad_h_x * scale
            v_safe_y = -grad_h_y * scale
        vfx, vfy = vdes_x + v_safe_x, vdes_y + v_safe_y
        intervened = True
    else:
        vfx, vfy = vdes_x, vdes_y
        intervened = False

    hdot = grad_h_x * vfx + grad_h_y * vfy
    constraint_lhs = hdot + p.alpha * h
    constraint_ok = constraint_lhs >= -constraint_tol
    away_x, away_y = -cx / rho, -cy / rho
    vsafe_align = (vfx - vdes_x) * away_x + (vfy - vdes_y) * away_y
    grad_h_norm = math.hypot(grad_h_x, grad_h_y)
    return CBFResult(vfx, vfy, intervened, rho, h, psi,
                     constraint_lhs, constraint_ok, vsafe_align, grad_h_norm)


# -------------------------------------------------------------------------------------------------
# Distance-barrier filter — single closest point, closed form. UNCHANGED.
# -------------------------------------------------------------------------------------------------
def cbf_filter_distance(vdes_x, vdes_y, closest_xy, p: CBFParams, constraint_tol: float = 1e-6) -> CBFResult:
    if closest_xy is None:
        return CBFResult(vdes_x, vdes_y, False, None, None, None, None, None, None, None)
    cx, cy = closest_xy
    rho = math.hypot(cx, cy)
    if rho == 0.0:
        return CBFResult(0.0, 0.0, True, 0.0, None, None, None, None, None, None)

    h = rho - p.d_safe
    grad_h_x = -cx / rho
    grad_h_y = -cy / rho
    psi = (grad_h_x * vdes_x + grad_h_y * vdes_y) + p.alpha * h
    if psi < 0.0:
        grad_h_sq = grad_h_x ** 2 + grad_h_y ** 2
        if grad_h_sq < 1e-12:
            v_safe_x = v_safe_y = 0.0
        else:
            scale = psi / grad_h_sq
            v_safe_x = -grad_h_x * scale
            v_safe_y = -grad_h_y * scale
        vfx, vfy = vdes_x + v_safe_x, vdes_y + v_safe_y
        intervened = True
    else:
        vfx, vfy = vdes_x, vdes_y
        intervened = False

    hdot = grad_h_x * vfx + grad_h_y * vfy
    constraint_lhs = hdot + p.alpha * h
    constraint_ok = constraint_lhs >= -constraint_tol
    away_x, away_y = -cx / rho, -cy / rho
    vsafe_align = (vfx - vdes_x) * away_x + (vfy - vdes_y) * away_y
    grad_h_norm = math.hypot(grad_h_x, grad_h_y)
    return CBFResult(vfx, vfy, intervened, rho, h, psi,
                     constraint_lhs, constraint_ok, vsafe_align, grad_h_norm)


# -------------------------------------------------------------------------------------------------
# NEW: Multi-obstacle distance-barrier CBF via QP.
#   Enforces grad_h_i . v >= -alpha h_i for ALL obstacle points at once. Lets the robot thread
#   between obstacles (correcting away from one cannot violate another). Pass ANY list of (x,y)
#   obstacle points: one-per-cluster (closestPoints), K-nearest, or all-points-in-radius.
# -------------------------------------------------------------------------------------------------
def cbf_filter_multi(vdes_x, vdes_y, obstacle_points, p: CBFParams,
                     constraint_tol: float = 1e-6) -> CBFResult:
    """
    QP:  min ||v - v_des||^2   s.t.  grad_h_i . v >= -alpha h_i  for every obstacle i.
    quadprog form  min 1/2 v^T G v - a^T v  s.t. C^T v >= b :
        G = 2I,  a = 2 v_des,  C[:,i] = grad_h_i (unit),  b[i] = -alpha h_i,  h_i = rho_i - d_safe.
    """
    vdes = np.array([float(vdes_x), float(vdes_y)])

    # No obstacles -> pass through.
    if not obstacle_points:
        return CBFResult(vdes_x, vdes_y, False, None, None, None, None, None, None, None,
                         n_obstacles=0, n_active=0)

    cols, b_list, diag = [], [], []
    for (cx, cy) in obstacle_points:
        rho = math.hypot(cx, cy)
        if rho < 1e-9:
            # Obstacle on the robot: no direction defined -> stop translating.
            return CBFResult(0.0, 0.0, True, 0.0, None, None, None, None, None, None,
                             n_obstacles=len(obstacle_points), n_active=None)
        gx, gy = -cx / rho, -cy / rho           # grad_h_i, unit (|grad_h| = 1)
        h = rho - p.d_safe
        cols.append([gx, gy])
        b_list.append(-p.alpha * h)
        diag.append((rho, h, gx, gy))

    G = 2.0 * np.eye(2)
    a = 2.0 * vdes
    C = np.array(cols).T                         # (2 x N): each column one constraint
    b = np.array(b_list)

    try:
        sol = quadprog.solve_qp(G, a, C, b, 0)
        v = sol[0]
    except ValueError:
        # Mutually infeasible constraints (boxed in on all sides) -> safest action: stop.
        return CBFResult(0.0, 0.0, True, None, None, None, None, False, None, None,
                         n_obstacles=len(diag), n_active=None)

    vfx, vfy = float(v[0]), float(v[1])

    # Diagnostics: nearest rho / min h, binding-constraint count, feasibility check.
    rho_min = min(d[0] for d in diag)
    h_min = min(d[1] for d in diag)
    n_active = 0
    constraint_ok = True
    for (rho, h, gx, gy) in diag:
        lhs = gx * vfx + gy * vfy + p.alpha * h          # >= -tol for every constraint
        if lhs < -constraint_tol:
            constraint_ok = False
        if abs((gx * vfx + gy * vfy) - (-p.alpha * h)) < 1e-4:
            n_active += 1

    intervened = not (abs(vfx - vdes_x) < 1e-9 and abs(vfy - vdes_y) < 1e-9)

    return CBFResult(vfx, vfy, intervened, rho_min, h_min, None,
                     None, constraint_ok, None, 1.0,
                     n_obstacles=len(diag), n_active=n_active)


# -------------------------------------------------------------------------------------------------
# NEW: Multi-obstacle distance-barrier CBF with a SOFT (slacked) constraint, via QP.
#
#   Each barrier is relaxed by a single shared slack epsilon >= 0:
#        grad_h_i . v + eps >= -alpha h_i
#   penalized by p_slack * eps^2 in the cost. The QP uses as little slack as possible, so eps stays
#   0 whenever the hard constraints are satisfiable and lifts only when the robot would otherwise be
#   boxed in. This removes the hard-stop infeasibility cliff: the robot passes GENTLY through tight
#   gaps instead of halting.
#
#   SAFETY NOTE: eps > 0 means h is allowed to go negative (the formal "never unsafe" guarantee is
#   traded for feasibility). Keep d_safe strictly below the true collision distance so a slack
#   excursion eats margin, not the bumper, and MONITOR eps (large/persistent eps => raise p_slack
#   or lower d_safe).
#
#   Variables solved: x = [vx, vy, eps].   x = [vx, vy, eps].
#   QP (quadprog form  min 1/2 x^T G x - a^T x  s.t. C^T x >= b):
#       G = diag(2, 2, 2*p_slack)        # velocity cost + slack penalty
#       a = [2 vdes_x, 2 vdes_y, 0]      # slack has no linear cost term
#       obstacle column i = [gx_i, gy_i, 1]   with  b_i = -alpha h_i   (softened by +eps)
#       slack-nonneg column = [0, 0, 1]       with  b   = 0            (enforces eps >= 0)
# -------------------------------------------------------------------------------------------------
def cbf_filter_multi_slack(vdes_x, vdes_y, obstacle_points, p: CBFParams,
                           constraint_tol: float = 1e-6) -> CBFResult:
    vdes = np.array([float(vdes_x), float(vdes_y)])

    if not obstacle_points:
        return CBFResult(vdes_x, vdes_y, False, None, None, None, None, None, None, None,
                         n_obstacles=0, n_active=0, eps=0.0)

    # First constraint column/entry enforces eps >= 0:  0*vx + 0*vy + 1*eps >= 0
    cols   = [[0.0, 0.0, 1.0]]
    b_list = [0.0]
    diag   = []

    for (cx, cy) in obstacle_points:
        rho = math.hypot(cx, cy)
        if rho < 1e-9:
            return CBFResult(0.0, 0.0, True, 0.0, None, None, None, None, None, None,
                             n_obstacles=len(obstacle_points), n_active=None, eps=None)
        gx, gy = -cx / rho, -cy / rho
        h = rho - p.d_safe
        cols.append([gx, gy, 1.0])             # softened: grad_h . v + eps >= -alpha h
        b_list.append(-p.alpha * h)
        diag.append((rho, h, gx, gy))

    G = np.diag([2.0, 2.0, 2.0 * p.p_slack])    # (vx,vy) cost + slack penalty
    a = np.array([2.0 * vdes[0], 2.0 * vdes[1], 0.0])
    C = np.array(cols).T                         # (3 x M): 3 vars, M = N obstacles + 1 (eps>=0)
    b = np.array(b_list)

    try:
        sol = quadprog.solve_qp(G, a, C, b, 0)   # all inequalities -> meq=0
        x = sol[0]
    except ValueError:
        # With slack the QP is essentially always feasible; this should be rare. Stop if it happens.
        return CBFResult(0.0, 0.0, True, None, None, None, None, False, None, None,
                         n_obstacles=len(diag), n_active=None, eps=None)

    vfx, vfy, eps = float(x[0]), float(x[1]), float(x[2])
    eps = max(0.0, eps)   # numerical guard: tiny negative -> 0

    rho_min = min(d[0] for d in diag)
    h_min   = min(d[1] for d in diag)

    # Constraint accounting against the SOFTENED constraint (grad_h . v + eps >= -alpha h).
    n_active = 0
    constraint_ok = True
    for (rho, h, gx, gy) in diag:
        soft_lhs = gx * vfx + gy * vfy + eps + p.alpha * h     # >= -tol for the soft constraint
        if soft_lhs < -constraint_tol:
            constraint_ok = False
        if abs((gx * vfx + gy * vfy + eps) - (-p.alpha * h)) < 1e-4:
            n_active += 1

    intervened = not (abs(vfx - vdes_x) < 1e-9 and abs(vfy - vdes_y) < 1e-9)

    return CBFResult(vfx, vfy, intervened, rho_min, h_min, None,
                     None, constraint_ok, None, 1.0,
                     n_obstacles=len(diag), n_active=n_active, eps=eps)


# -------------------------------------------------------------------------------------------------
# NEW: Multi-obstacle distance-barrier CBF with a LINEAR (L1-style) shared slack penalty.
#
#   Same softened constraints (grad_h_i . v + eps >= -alpha h_i, eps >= 0), but the cost penalizes
#   slack LINEARLY:  +p_slack_lin * eps   (instead of quadratically). Effect: a THRESHOLD/bang-bang
#   relaxation -- eps stays exactly 0 until constraint pressure (sum of multipliers) exceeds
#   p_slack_lin, then opens up. Crisper "hard until forced" behavior vs the smooth quadratic spring.
#
#   quadprog needs G positive-DEFINITE; a pure linear slack term gives a 0 on the slack diagonal
#   (only semi-definite). We add a tiny quadratic regularizer (reg) on slack to keep it solvable.
#   variables x = [vx, vy, eps]:
#       G = diag(2, 2, 2*reg)                 # reg is small; slack curvature ~ 0
#       a = [2 vdes_x, 2 vdes_y, -p_slack_lin] # quadprog minimizes -a^T x => +p_slack_lin*eps
#       obstacle col i = [gx_i, gy_i, 1], b_i = -alpha h_i ;  slack col = [0,0,1], b = 0
# -------------------------------------------------------------------------------------------------
def cbf_filter_multi_slack_linear(vdes_x, vdes_y, obstacle_points, p: CBFParams,
                                  constraint_tol: float = 1e-6, reg: float = 1e-3) -> CBFResult:
    vdes = np.array([float(vdes_x), float(vdes_y)])

    if not obstacle_points:
        return CBFResult(vdes_x, vdes_y, False, None, None, None, None, None, None, None,
                         n_obstacles=0, n_active=0, eps=0.0)

    cols   = [[0.0, 0.0, 1.0]]      # eps >= 0
    b_list = [0.0]
    diag   = []
    for (cx, cy) in obstacle_points:
        rho = math.hypot(cx, cy)
        if rho < 1e-9:
            return CBFResult(0.0, 0.0, True, 0.0, None, None, None, None, None, None,
                             n_obstacles=len(obstacle_points), n_active=None, eps=None)
        gx, gy = -cx / rho, -cy / rho
        h = rho - p.d_safe
        cols.append([gx, gy, 1.0])
        b_list.append(-p.alpha * h)
        diag.append((rho, h, gx, gy))

    G = np.diag([2.0, 2.0, 2.0 * reg])                       # tiny slack curvature (PD-safe)
    a = np.array([2.0 * vdes[0], 2.0 * vdes[1], -p.p_slack_lin])  # linear slack cost
    C = np.array(cols).T
    b = np.array(b_list)

    try:
        sol = quadprog.solve_qp(G, a, C, b, 0)
        x = sol[0]
    except ValueError:
        return CBFResult(0.0, 0.0, True, None, None, None, None, False, None, None,
                         n_obstacles=len(diag), n_active=None, eps=None)

    vfx, vfy, eps = float(x[0]), float(x[1]), max(0.0, float(x[2]))
    rho_min = min(d[0] for d in diag)
    h_min   = min(d[1] for d in diag)
    n_active = 0
    constraint_ok = True
    for (rho, h, gx, gy) in diag:
        if (gx * vfx + gy * vfy + eps + p.alpha * h) < -constraint_tol:
            constraint_ok = False
        if abs((gx * vfx + gy * vfy + eps) - (-p.alpha * h)) < 1e-4:
            n_active += 1
    intervened = not (abs(vfx - vdes_x) < 1e-9 and abs(vfy - vdes_y) < 1e-9)
    return CBFResult(vfx, vfy, intervened, rho_min, h_min, None,
                     None, constraint_ok, None, 1.0,
                     n_obstacles=len(diag), n_active=n_active, eps=eps)


# -------------------------------------------------------------------------------------------------
# NEW: Multi-obstacle distance-barrier CBF with DISTANCE-WEIGHTED per-obstacle slack.
#
#   Each obstacle gets its OWN slack eps_i >= 0, penalized QUADRATICALLY with a weight that grows as
#   the obstacle nears:   sum_i  p_i * eps_i^2,   p_i = p_base / (h_i + eta_w)^2.
#   => far obstacle (large h): small p_i, cheap to relax (gentle, there's room).
#   => near obstacle (h -> 0): huge p_i, that constraint becomes effectively HARD again (protects
#      the body automatically; the relaxation self-extinguishes at the barrier).
#
#   variables x = [vx, vy, eps_1, ..., eps_N]   (2 + N).
#       G = diag(2, 2, 2 p_1, ..., 2 p_N)
#       a = [2 vdes_x, 2 vdes_y, 0, ..., 0]
#       obstacle i constraint:  grad_h_i . v + eps_i >= -alpha h_i
#           => column has [gx_i, gy_i] in the velocity rows and a single 1 in ITS OWN eps_i row.
#       eps_i >= 0  for each i  => identity block on the slack rows.
# -------------------------------------------------------------------------------------------------
def cbf_filter_multi_slack_weighted(vdes_x, vdes_y, obstacle_points, p: CBFParams,
                                    constraint_tol: float = 1e-6) -> CBFResult:
    vdes = np.array([float(vdes_x), float(vdes_y)])

    if not obstacle_points:
        return CBFResult(vdes_x, vdes_y, False, None, None, None, None, None, None, None,
                         n_obstacles=0, n_active=0, eps=0.0)

    diag = []
    for (cx, cy) in obstacle_points:
        rho = math.hypot(cx, cy)
        if rho < 1e-9:
            return CBFResult(0.0, 0.0, True, 0.0, None, None, None, None, None, None,
                             n_obstacles=len(obstacle_points), n_active=None, eps=None)
        gx, gy = -cx / rho, -cy / rho
        h = rho - p.d_safe
        diag.append((rho, h, gx, gy))

    N = len(diag)
    nvars = 2 + N

    # Cost: velocity (2,2) + per-obstacle slack weights p_i on the diagonal.
    Gdiag = [2.0, 2.0]
    for (rho, h, gx, gy) in diag:
        p_i = p.p_base / (h + p.eta_w) ** 2     # near obstacle (small/neg h) -> large weight
        Gdiag.append(2.0 * p_i)
    G = np.diag(Gdiag)
    a = np.array([2.0 * vdes[0], 2.0 * vdes[1]] + [0.0] * N)

    # Constraints. Columns of C (each length nvars):
    cols, b_list = [], []
    # (a) softened barriers: grad_h_i . v + eps_i >= -alpha h_i
    for i, (rho, h, gx, gy) in enumerate(diag):
        col = [gx, gy] + [0.0] * N
        col[2 + i] = 1.0                        # eps_i couples only into ITS OWN constraint
        cols.append(col)
        b_list.append(-p.alpha * h)
    # (b) eps_i >= 0 for each i
    for i in range(N):
        col = [0.0, 0.0] + [0.0] * N
        col[2 + i] = 1.0
        cols.append(col)
        b_list.append(0.0)

    C = np.array(cols).T                         # (nvars x (2N))
    b = np.array(b_list)

    try:
        sol = quadprog.solve_qp(G, a, C, b, 0)
        x = sol[0]
    except ValueError:
        return CBFResult(0.0, 0.0, True, None, None, None, None, False, None, None,
                         n_obstacles=N, n_active=None, eps=None)

    vfx, vfy = float(x[0]), float(x[1])
    eps_vec = [max(0.0, float(x[2 + i])) for i in range(N)]
    eps_max = max(eps_vec) if eps_vec else 0.0   # report the largest per-obstacle slack

    rho_min = min(d[0] for d in diag)
    h_min   = min(d[1] for d in diag)
    n_active = 0
    constraint_ok = True
    for i, (rho, h, gx, gy) in enumerate(diag):
        ei = eps_vec[i]
        if (gx * vfx + gy * vfy + ei + p.alpha * h) < -constraint_tol:
            constraint_ok = False
        if abs((gx * vfx + gy * vfy + ei) - (-p.alpha * h)) < 1e-4:
            n_active += 1
    intervened = not (abs(vfx - vdes_x) < 1e-9 and abs(vfy - vdes_y) < 1e-9)
    return CBFResult(vfx, vfy, intervened, rho_min, h_min, None,
                     None, constraint_ok, None, 1.0,
                     n_obstacles=N, n_active=n_active, eps=eps_max)


# -------------------------------------------------------------------------------------------------
# NEW: Multi-obstacle distance-barrier CBF with ELASTIC-NET shared slack (L1 + L2 combined).
#
#   Penalty:  p_elastic_lin * eps  +  p_elastic_quad * eps^2     (with eps >= 0)
#       - the LINEAR (L1) term creates a DEAD-ZONE: slack stays exactly 0 until constraint pressure
#         exceeds p_elastic_lin (stay hard over a wide range, like the pure-linear mode),
#       - the QUADRATIC (L2) term makes the response SMOOTH once slack DOES engage (gentle easing,
#         like the pure-quadratic mode).
#   So this is the deliberate combination of the linear and quadratic modes: "hard until forced,
#   then ease gently." Unlike the regularizer 'reg' in the pure-linear filter (a tiny solver hack),
#   here BOTH terms are meaningful, tunable design parameters.
#
#   variables x = [vx, vy, eps]:
#       G = diag(2, 2, 2 * p_elastic_quad)              # L2 term -> real slack curvature
#       a = [2 vdes_x, 2 vdes_y, -p_elastic_lin]        # L1 term (quadprog minimizes -a^T x => +lin*eps)
#       obstacle col i = [gx_i, gy_i, 1], b_i = -alpha h_i ;  slack col = [0,0,1], b = 0
# -------------------------------------------------------------------------------------------------
def cbf_filter_multi_slack_elastic(vdes_x, vdes_y, obstacle_points, p: CBFParams,
                                   constraint_tol: float = 1e-6) -> CBFResult:
    vdes = np.array([float(vdes_x), float(vdes_y)])

    if not obstacle_points:
        return CBFResult(vdes_x, vdes_y, False, None, None, None, None, None, None, None,
                         n_obstacles=0, n_active=0, eps=0.0)

    cols   = [[0.0, 0.0, 1.0]]      # eps >= 0
    b_list = [0.0]
    diag   = []
    for (cx, cy) in obstacle_points:
        rho = math.hypot(cx, cy)
        if rho < 1e-9:
            return CBFResult(0.0, 0.0, True, 0.0, None, None, None, None, None, None,
                             n_obstacles=len(obstacle_points), n_active=None, eps=None)
        gx, gy = -cx / rho, -cy / rho
        h = rho - p.d_safe
        cols.append([gx, gy, 1.0])
        b_list.append(-p.alpha * h)
        diag.append((rho, h, gx, gy))

    # L2 term sets slack curvature (must be > 0 for quadprog PD-ness); L1 term is the linear cost.
    quad = p.p_elastic_quad if p.p_elastic_quad > 1e-9 else 1e-3   # guard PD if user zeros it
    G = np.diag([2.0, 2.0, 2.0 * quad])
    a = np.array([2.0 * vdes[0], 2.0 * vdes[1], -p.p_elastic_lin])
    C = np.array(cols).T
    b = np.array(b_list)

    try:
        sol = quadprog.solve_qp(G, a, C, b, 0)
        x = sol[0]
    except ValueError:
        return CBFResult(0.0, 0.0, True, None, None, None, None, False, None, None,
                         n_obstacles=len(diag), n_active=None, eps=None)

    vfx, vfy, eps = float(x[0]), float(x[1]), max(0.0, float(x[2]))
    rho_min = min(d[0] for d in diag)
    h_min   = min(d[1] for d in diag)
    n_active = 0
    constraint_ok = True
    for (rho, h, gx, gy) in diag:
        if (gx * vfx + gy * vfy + eps + p.alpha * h) < -constraint_tol:
            constraint_ok = False
        if abs((gx * vfx + gy * vfy + eps) - (-p.alpha * h)) < 1e-4:
            n_active += 1
    intervened = not (abs(vfx - vdes_x) < 1e-9 and abs(vfy - vdes_y) < 1e-9)
    return CBFResult(vfx, vfy, intervened, rho_min, h_min, None,
                     None, constraint_ok, None, 1.0,
                     n_obstacles=len(diag), n_active=n_active, eps=eps)


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
            f"  vfinal: ({r.vx:+.3f}, {r.vy:+.3f})")


# ============================ self-test ============================
if __name__ == '__main__':
    p = CBFParams(alpha=1.0, d_safe=0.4, p_slack=50.0, p_slack_lin=5.0, p_base=5.0, eta_w=0.05)

    def row(name, r):
        eps = f"{r.eps:.4f}" if r.eps is not None else " STOP"
        print(f"    {name:18s} v=({r.vx:+.3f},{r.vy:+.3f})  eps={eps}  active={r.n_active}")

    scenarios = [
        ("wide gap (feasible)",      0.3, 0.0, [(0.5,0.6),(0.5,-0.6)]),
        ("narrow gap (feasible)",    0.3, 0.0, [(0.3,0.35),(0.3,-0.35)]),
        ("tight gap (< 2 d_safe)",   0.3, 0.0, [(0.3,0.25),(0.3,-0.25)]),
        ("boxed in (infeasible)",    0.3, 0.0, [(0.2,0.0),(-0.2,0.0),(0.0,0.2),(0.0,-0.2)]),
        ("ASYMMETRIC: L near,R far", 0.3, 0.0, [(0.35,0.18),(0.5,-0.6)]),
    ]
    for name, vx, vy, obs in scenarios:
        print(f"\n--- {name}  (obstacles={len(obs)}) ---")
        row("quadratic", cbf_filter_multi_slack(vx, vy, obs, p))
        row("linear",    cbf_filter_multi_slack_linear(vx, vy, obs, p))
        row("weighted",  cbf_filter_multi_slack_weighted(vx, vy, obs, p))
        row("elastic",   cbf_filter_multi_slack_elastic(vx, vy, obs, p))

    print("\n--- linear threshold demo: sweep p_slack_lin on the tight gap ---")
    for pl in (0.5, 2.0, 5.0, 20.0):
        pp = CBFParams(alpha=1.0, d_safe=0.4, p_slack_lin=pl)
        r = cbf_filter_multi_slack_linear(0.3, 0.0, [(0.3,0.25),(0.3,-0.25)], pp)
        print(f"    p_slack_lin={pl:5.1f}: v=({r.vx:+.3f},{r.vy:+.3f}) eps={r.eps:.4f}")

    print("\n--- elastic dead-zone demo: increasing constraint pressure (tighter gaps) ---")
    print("    fixed p_elastic_lin=5, p_elastic_quad=50; gap shrinks -> pressure rises")
    pe = CBFParams(alpha=1.0, d_safe=0.4, p_elastic_lin=5.0, p_elastic_quad=50.0)
    for lat in (0.55, 0.45, 0.38, 0.32, 0.28):
        r = cbf_filter_multi_slack_elastic(0.3, 0.0, [(0.3,lat),(0.3,-lat)], pe)
        gap = 2*lat
        print(f"    gap={gap:.2f}m (h_min={r.h:+.3f}): v=({r.vx:+.3f},{r.vy:+.3f}) eps={r.eps:.4f}")
