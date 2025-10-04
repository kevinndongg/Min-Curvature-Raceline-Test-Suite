import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d 

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

def compute_midline(left_x, left_y, right_x, right_y, num_points=100):
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

    # Resample to equally spaced points
    mid_x, mid_y = resample_equal_distance(midline[:,0], midline[:,1], num_points)

    return mid_x, mid_y

if __name__ == "__main__":
    df = pd.read_csv('./tracks/' + INPUT_FILE_NAME)

    left_df = df[df['type'] == 'left']
    right_df = df[df['type'] == 'right']

    left_x = left_df['x'].to_numpy()
    left_y = left_df['y'].to_numpy()
    right_x = right_df['x'].to_numpy()
    right_y = right_df['y'].to_numpy()

    mid_x, mid_y = compute_midline(left_x, left_y, right_x, right_y)
    
    plt.figure(figsize=(10,6))
    plt.plot(left_x, left_y, color='blue', marker='o')
    plt.plot(right_x, right_y, color='gold', marker='o')
    plt.plot(mid_x, mid_y, color='green', marker='x')
    plt.grid(True)
    plt.show()