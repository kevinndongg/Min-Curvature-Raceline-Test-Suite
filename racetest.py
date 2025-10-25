import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d, splprep, splev

INPUT_FILE_NAME = "test_track.csv" # Modify this to be your track!

def order_path(points):
    """Greedy nearest-neighbor path ordering of points."""
    points = np.array(points)
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    path = [0]  # start at first point
    visited[0] = True

    for _ in range(n - 1):
        last = points[path[-1]]
        dists = np.linalg.norm(points - last, axis=1)
        dists[visited] = np.inf
        next_idx = np.argmin(dists)
        path.append(next_idx)
        visited[next_idx] = True
    return path

def resample_equal_distance(x, y, num_points=100):
    """Resample curve (x,y) so points are equally spaced along arc length."""
    pts = np.column_stack((x, y))
    # Distances between consecutive points
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate(([0], np.cumsum(dists)))
    total_length = arc[-1]
    target_arc = np.linspace(0, total_length, num_points)

    # Interpolators
    fx = interp1d(arc, x)
    fy = interp1d(arc, y)

    x_new = fx(target_arc)
    y_new = fy(target_arc)
    return x_new, y_new

def compute_midline_near_neighbor(left_x, left_y, right_x, right_y, num_points=100):
    """
    Compute equally spaced midline points given left and right cones.
    """
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

    # Order points into a connected path
    order = order_path(midline)
    midline = midline[order]

    return midline[:, 0], midline[:,1]

def fit_b_spline(x, y):
    # Fit B-spline (s=0 → exact fit, per=False → open curve)
    tck, u = splprep([x, y], s=3, per=1)

    # Sample spline
    u_new = np.linspace(0, 1, 400)     # dense sampling
    x_new, y_new = splev(u_new, tck)

    return x_new, y_new, tck

def remove_duplicates_preserve_order(x, y):
    seen = set()
    new_x = []
    new_y = []
    for xi, yi in zip(x, y):
        if (xi, yi) not in seen:
            seen.add((xi, yi))
            new_x.append(xi)
            new_y.append(yi)
    return np.array(new_x), np.array(new_y)

from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

def sample_spline_constant_distance(tck, num_samples=1000, spacing=3.0):
    """
    Sample B-spline at roughly constant `spacing` distance (e.g., 3 m apart).
    """
    # Dense parameter sampling
    t_dense = np.linspace(0, 1, num_samples)
    x_dense, y_dense = splev(t_dense, tck)
    dx, dy = splev(t_dense, tck, der=1)
    
    # Compute incremental arc length
    ds = np.sqrt(dx**2 + dy**2)
    arc_length = cumulative_trapezoid(ds, t_dense, initial=0)
    total_length = arc_length[-1]
    
    # Target arc-length positions every `spacing` meters
    num_waypoints = int(total_length // spacing)
    target_arc = np.linspace(0, num_waypoints * spacing, num_waypoints + 1)

    # Interpolate to get corresponding t_i values
    f_t = interp1d(arc_length, t_dense)
    t_samples = f_t(target_arc)

    # Evaluate spline at sampled t_i
    x_samples, y_samples = splev(t_samples, tck)
    return x_samples, y_samples, total_length

def sample_spline_constant_distance_with_curvature(tck, num_samples=1000, spacing=3.0):
    """
    Sample B-spline at roughly constant spacing and compute curvature at each point.
    Returns (x, y, curvature, total_length)
    """
    # Dense sampling for integration
    t_dense = np.linspace(0, 1, num_samples)
    dx, dy = splev(t_dense, tck, der=1)
    ddx, ddy = splev(t_dense, tck, der=2)
    
    # Arc length integration
    ds = np.sqrt(dx**2 + dy**2)
    arc_length = cumulative_trapezoid(ds, t_dense, initial=0)
    total_length = arc_length[-1]

    # Target arc-length positions
    num_waypoints = int(total_length // spacing)
    target_arc = np.linspace(0, num_waypoints * spacing, num_waypoints + 1)
    f_t = interp1d(arc_length, t_dense)
    t_samples = f_t(target_arc)

    # Evaluate spline at waypoints
    x, y = splev(t_samples, tck)
    dx, dy = splev(t_samples, tck, der=1)
    ddx, ddy = splev(t_samples, tck, der=2)

    # Compute curvature
    curvature = np.abs(dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    return x, y, curvature, total_length

if __name__ == "__main__":
    df = pd.read_csv('./tracks/' + INPUT_FILE_NAME)

    left_df = df[df['type'] == 'left']
    right_df = df[df['type'] == 'right']

    left_x = left_df['x'].to_numpy()
    left_y = left_df['y'].to_numpy()
    right_x = right_df['x'].to_numpy()
    right_y = right_df['y'].to_numpy()

    mid_x, mid_y = compute_midline_near_neighbor(left_x, left_y, right_x, right_y)
    mid_eq_dist_x, mid_eq_dist_y = resample_equal_distance(mid_x, mid_y, 100)

    print(len(mid_x), len(mid_y))
    print(type(mid_x), type(mid_y))
    print(mid_x.shape, mid_y.shape)
    print(len(np.unique(mid_x)), len(np.unique(mid_y)))

    no_dupes_x, no_dupes_y = remove_duplicates_preserve_order(mid_x, mid_y);

    x_spline, y_spline, tck = fit_b_spline(no_dupes_x, no_dupes_y)

    x_const, y_const, curvature, total_length = sample_spline_constant_distance_with_curvature(tck, spacing=3.0)

    print(f"Total track length ≈ {total_length:.2f} m")
    print(f"Number of 3m waypoints: {len(x_const)}")
    
    plt.figure(figsize=(10,6))
    plt.scatter(x_const, y_const, c=curvature, cmap='viridis', s=25)
    plt.plot(left_x, left_y, color='blue', marker='o')
    plt.plot(right_x, right_y, color='gold', marker='o')
    # plt.plot(mid_x, mid_y, color='green', marker='x')
    plt.plot(x_spline, y_spline, '-', color='purple', label='B-spline')  # spline
    # plt.plot(mid_eq_dist_x, mid_eq_dist_y, color='pink', marker = '*')
    # plt.plot(x_const, y_const, 'r.-', label='3m spaced waypoints')
    plt.grid(True)
    plt.show()