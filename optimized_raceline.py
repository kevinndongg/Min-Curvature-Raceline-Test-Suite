# optimized_raceline.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d, splprep, splev, BSpline
from scipy.integrate import cumulative_trapezoid
import cvxpy as cp

INPUT_FILE_NAME = "silverstone.csv"  # adjust path if needed
SPACING = 3.0    # meters between samples
DENSE_SAMPLES = 2000  # for arc-length estimate
SPLINE_SAMPLES_PLOT = 400
SAFETY_MARGIN = 1.0  # meters you want to stay away from cones

# ----- helper functions from your snippet (slightly adapted) -----
def order_path(points):
    points = np.array(points)
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    path = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = points[path[-1]]
        dists = np.linalg.norm(points - last, axis=1)
        dists[visited] = np.inf
        next_idx = np.argmin(dists)
        path.append(next_idx)
        visited[next_idx] = True
    return path

def compute_midline_near_neighbor(left_x, left_y, right_x, right_y):
    left_cones = np.column_stack((left_x, left_y))
    right_cones = np.column_stack((right_x, right_y))

    tree_left = cKDTree(left_cones)
    tree_right = cKDTree(right_cones)

    midline_points = []
    for p in left_cones:
        _, idx = tree_right.query(p)
        midpoint = (p + right_cones[idx]) / 2.0
        midline_points.append(midpoint)
    for p in right_cones:
        _, idx = tree_left.query(p)
        midpoint = (p + left_cones[idx]) / 2.0
        midline_points.append(midpoint)
    midline = np.array(midline_points)
    order = order_path(midline)
    midline = midline[order]
    return midline[:, 0], midline[:, 1]

def fit_periodic_b_spline(x, y, s=3):
    # returns tck and a dense sampled spline for plotting
    tck, u = splprep([x, y], s=s, per=1)
    u_new = np.linspace(0, 1, SPLINE_SAMPLES_PLOT)
    x_new, y_new = splev(u_new, tck)
    return tck, x_new, y_new

def sample_spline_constant_distance_with_t_and_derivs(tck, spacing=SPACING, dense_samples=DENSE_SAMPLES):
    # dense param sampling for arc-length
    t_dense = np.linspace(0, 1, dense_samples)
    dx_dense, dy_dense = splev(t_dense, tck, der=1)
    speed_dense = np.sqrt(dx_dense**2 + dy_dense**2)
    arc_dense = cumulative_trapezoid(speed_dense, t_dense, initial=0)
    total_length = arc_dense[-1]

    n_waypoints = max(1, int(total_length // spacing))
    target_arc = np.linspace(0, n_waypoints * spacing, n_waypoints + 1)
    f_t = interp1d(arc_dense, t_dense)
    t_samples = f_t(target_arc)

    # evaluate at these sample params
    x_samples, y_samples = splev(t_samples, tck)
    dx, dy = splev(t_samples, tck, der=1)
    ddx, ddy = splev(t_samples, tck, der=2)

    return {
        "t_samples": t_samples,
        "x": np.array(x_samples),
        "y": np.array(y_samples),
        "dx": np.array(dx),
        "dy": np.array(dy),
        "ddx": np.array(ddx),
        "ddy": np.array(ddy),
        "total_length": total_length,
        "ds": spacing
    }

# ---- QP helper: construct basis evaluation matrices ----
def build_basis_matrices_from_tck(tck, t_samples):
    # tck: (t, c, k)
    knots, coeffs, k = tck
    # number of control points (coeff arrays length)
    S = len(coeffs[0])
    M = len(t_samples)

    # We'll build the basis evaluation matrices by creating S BSpline objects
    # where each has coefficients e_j (1 at j).  Then evaluate BSpline and its 2nd derivative.
    B = np.zeros((M, S))    # basis -> positions (for x or y)
    B2 = np.zeros((M, S))   # second derivative of basis

    # SciPy BSpline expects a full knot vector for the periodic case splprep returns,
    # so we can reuse knots and degree k for each basis function.
    for j in range(S):
        coeff_vec = np.zeros(S)
        coeff_vec[j] = 1.0
        bs = BSpline(knots, coeff_vec, k, extrapolate=False)
        # basis value (position) and second derivative at t_samples
        B[:, j] = bs(t_samples)
        B2[:, j] = bs.derivative(2)(t_samples)
    return B, B2

# ---- main optimization routine ----
def optimize_raceline(tck, left_x, left_y, right_x, right_y,
                      spacing=SPACING, verbose=True):
    # sample spline & derivatives
    samp = sample_spline_constant_distance_with_t_and_derivs(tck, spacing=spacing)
    t_samples = samp["t_samples"]
    x_s = samp["x"]; y_s = samp["y"]
    dx = samp["dx"]; dy = samp["dy"]
    ddx = samp["ddx"]; ddy = samp["ddy"]
    M = len(x_s)

    # compute tangent angles and normal vectors
    theta = np.arctan2(dy, dx)
    sin_th = np.sin(theta); cos_th = np.cos(theta)
    # left/right widths: nearest cone distance projections
    left_pts = np.column_stack((left_x, left_y))
    right_pts = np.column_stack((right_x, right_y))
    tree_left = cKDTree(left_pts)
    tree_right = cKDTree(right_pts)

    # For each sample, find nearest left cone & right cone and compute lateral distance
    _, idx_left = tree_left.query(np.column_stack((x_s, y_s)))
    _, idx_right = tree_right.query(np.column_stack((x_s, y_s)))
    left_near = left_pts[idx_left]
    right_near = right_pts[idx_right]
    # project vector from sample to cone onto normal to get lateral distance
    vec_to_left = left_near - np.column_stack((x_s, y_s))
    vec_to_right = right_near - np.column_stack((x_s, y_s))
    left_width = np.einsum('ij,ij->i', vec_to_left, np.column_stack(( -sin_th, cos_th )))
    right_width = np.einsum('ij,ij->i', vec_to_right, np.column_stack(( sin_th, -cos_th )))
    # ensure positive
    left_width = np.abs(left_width) + 0.01
    right_width = np.abs(right_width) + 0.01

    if verbose:
        print(f"Samples: {M}, total length ≈ {samp['total_length']:.2f} m")
        print(f"Average left width: {np.mean(left_width):.2f}, right width: {np.mean(right_width):.2f}")

    # Build basis matrices B and B2 (M x S)
    B, B2 = build_basis_matrices_from_tck(tck, t_samples)
    S = B.shape[1]

    # Build block matrices for x & y
    # B2 maps control points -> second derivative contribution to x (or y)
    # Build Bx = [B2  0] , By = [0  B2] (dimensions M x 2S)
    B2_block_x = np.hstack([B2, np.zeros_like(B2)])
    B2_block_y = np.hstack([np.zeros_like(B2), B2])

    # QP cost H = Bx^T Bx + By^T By  (this corresponds to minimizing L2 norm of second derivative magnitude)
    H = B2_block_x.T @ B2_block_x + B2_block_y.T @ B2_block_y
    # small regularizer to ensure PD
    H += 1e-8 * np.eye(H.shape[0])

    # Build linear constraints: for each sample i, lateral offset = (-sinθ_i * x_i + cosθ_i * y_i) - (-sinθ_i * x_mid + cosθ_i * y_mid)
    # But x_mid,y_mid are current midline points. We impose bounds on lateral offset relative to midline:
    # -right_width <= lateral_offset <= left_width
    # Express lateral_offset = r_i @ z  where z = [cx; cy]
    # For x_i = B_row @ cx, y_i = B_row @ cy
    R_rows = []
    lb = []
    ub = []
    # Build A_xrow = B_row for x, A_yrow = B_row for y (with appropriate zero blocks for stacking)
    for i in range(M):
        B_row = B[i:i+1, :]  # shape (1, S)
        A_x = np.hstack([B_row, np.zeros_like(B_row)])  # shape (1, 2S)
        A_y = np.hstack([np.zeros_like(B_row), B_row])
        # lateral projection row
        R_i = (-sin_th[i]) * A_x + (cos_th[i]) * A_y  # row vector shape (1,2S)
        R_rows.append(R_i)
        # bounds. We want: -right_width <= projected_point - projected_midline <= left_width
        # Note: projected_midline = (-sinθ * mid_x + cosθ * mid_y) which is equal to:
        mid_proj = -sin_th[i] * x_s[i] + cos_th[i] * y_s[i]
        # The linear constraint we build will be R_i @ z  (this directly computes projection of optimized spline point).
        # So we need lb_i = mid_proj - right_width[i], ub_i = mid_proj + left_width[i]
        left_bound = max(left_width[i] - SAFETY_MARGIN, 0.05)
        right_bound = max(right_width[i] - SAFETY_MARGIN, 0.05)
        lb.append(mid_proj - right_bound)
        ub.append(mid_proj + left_bound)

    R = np.vstack(R_rows)    # shape (M, 2S)
    lb = np.array(lb).reshape(-1)
    ub = np.array(ub).reshape(-1)

    # Build QP in cvxpy
    z = cp.Variable(2 * S)
    objective = 0.5 * cp.quad_form(z, H)
    constraints = [R @ z <= ub, R @ z >= lb]
    # Optionally limit how far control points can move from original (trust-region)
    # enforce |z - z0| <= delta per-control (small)
    # get z0 (initial control points flattened)
    _, c, k = tck
    cx0 = np.array(c[0])
    cy0 = np.array(c[1])
    z0 = np.concatenate([cx0, cy0])
    delta = 2.0  # meters, adjust if too tight/loose
    constraints += [z <= z0 + delta, z >= z0 - delta]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    print("Solving QP... (this may take a few seconds)")
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-4, eps_rel=1e-4, max_iter=100000)

    if z.value is None:
        raise RuntimeError("QP solver failed (no solution returned). Try loosening bounds or increasing regularization.")

    z_opt = z.value
    cx_opt = z_opt[:S]
    cy_opt = z_opt[S:]

    # build new tck and sample for plotting
    tks_opt = (tck[0], [cx_opt, cy_opt], tck[2])

    # compute curvature on optimized spline for visualization
    samp_opt = sample_spline_constant_distance_with_t_and_derivs(tks_opt, spacing=spacing)
    curv_opt = np.abs(samp_opt["dx"] * samp_opt["ddy"] - samp_opt["dy"] * samp_opt["ddx"]) / np.maximum((samp_opt["dx"]**2 + samp_opt["dy"]**2)**1.5, 1e-8)

    results = {
        "tck_opt": tks_opt,
        "cx_opt": cx_opt, "cy_opt": cy_opt,
        "x_mid": x_s, "y_mid": y_s,
        "x_opt_samples": samp_opt["x"], "y_opt_samples": samp_opt["y"],
        "curv_opt": curv_opt,
        "left_width": left_width, "right_width": right_width,
        "t_samples": t_samples,
    }
    return results

# ---- Integration: read CSV, run optimization, plot results ----
if __name__ == "__main__":
    df = pd.read_csv('./tracks/' + INPUT_FILE_NAME)

    left_df = df[df['type'] == 'left']
    right_df = df[df['type'] == 'right']

    left_x = left_df['x'].to_numpy()
    left_y = left_df['y'].to_numpy()
    right_x = right_df['x'].to_numpy()
    right_y = right_df['y'].to_numpy()

    mid_x, mid_y = compute_midline_near_neighbor(left_x, left_y, right_x, right_y)
    # remove duplicate adjacent points (very small edits)
    unique_idx = np.unique(np.round(np.column_stack((mid_x, mid_y)), 6), axis=0, return_index=True)[1]
    mid_x = mid_x[np.sort(unique_idx)]
    mid_y = mid_y[np.sort(unique_idx)]

    tck, mid_x_spline, mid_y_spline = fit_periodic_b_spline(mid_x, mid_y, s=3)

    # run optimization
    res = optimize_raceline(tck, left_x, left_y, right_x, right_y, spacing=SPACING, verbose=True)

    # plot
    plt.figure(figsize=(12, 8))
    # cones
    plt.scatter(left_x, left_y, c='blue', s=10, label='left cones')
    plt.scatter(right_x, right_y, c='gold', s=10, label='right cones')
    # midline and its control points
    cx0 = np.array(tck[1][0]); cy0 = np.array(tck[1][1])
    plt.plot(mid_x_spline, mid_y_spline, color='purple', label='initial midline spline')
    plt.plot(cx0, cy0, 'x--', color='purple', alpha=0.6, label='original ctrl pts')

    # optimized spline (dense)
    u_plot = np.linspace(0, 1, 600)
    x_opt_plot, y_opt_plot = splev(u_plot, res['tck_opt'])
    plt.plot(x_opt_plot, y_opt_plot, color='red', linewidth=2.0, label='optimized raceline')
    # optimized sample points colored by curvature
    plt.scatter(res['x_opt_samples'], res['y_opt_samples'], c=res['curv_opt'], cmap='viridis', s=25, label='opt samples (curv)')
    plt.colorbar(label='curvature (1/m)')

    plt.axis('equal')
    plt.legend()
    plt.title('Optimized raceline (minimize curvature^2 subject to track bounds)')
    plt.grid(True)
    plt.show()
