import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def find_midline(left_x, left_y, right_x, right_y, num_midpoints=None):
    """
    Given left and right boundary points, returns midline points by averaging.
    Handles different number of points by resampling via interpolation.

    Args:
        left_x, left_y: np.arrays of left boundary points
        right_x, right_y: np.arrays of right boundary points
        num_midpoints: number of points in the midline (default=max of left/right points)

    Returns:
        mid_x, mid_y: np.arrays of midline points
    """
    if num_midpoints is None:
        num_midpoints = max(len(left_x), len(right_x))

    # Parameterize boundaries from 0 to 1 by cumulative distance
    def parameterize_curve(x, y):
        dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cumdist = np.insert(np.cumsum(dist), 0, 0)
        return cumdist / cumdist[-1]

    t_left = parameterize_curve(left_x, left_y)
    t_right = parameterize_curve(right_x, right_y)

    # Interpolate left boundary to uniform t
    t_uniform = np.linspace(0, 1, num_midpoints)
    left_x_interp = np.interp(t_uniform, t_left, left_x)
    left_y_interp = np.interp(t_uniform, t_left, left_y)

    # Interpolate right boundary to uniform t
    right_x_interp = np.interp(t_uniform, t_right, right_x)
    right_y_interp = np.interp(t_uniform, t_right, right_y)

    # Average to get midline
    mid_x = (left_x_interp + right_x_interp) / 2
    mid_y = (left_y_interp + right_y_interp) / 2

    return mid_x, mid_y

if __name__ == "__main__":
    df = pd.read_csv('./tracks/ellipse_track.csv')

    left_df = df[df['type'] == 'left']
    right_df = df[df['type'] == 'right']

    left_x = left_df['x'].to_numpy()
    left_y = left_df['y'].to_numpy()
    right_x = right_df['x'].to_numpy()
    right_y = right_df['y'].to_numpy()

    mid_x, mid_y = find_midline(left_x, left_y, right_x, right_y)
    
    plt.figure(figsize=(10,6))
    plt.plot(left_x, left_y, color='blue', marker='o')
    plt.plot(right_x, right_y, color='gold', marker='o')
    plt.plot(mid_x, mid_y, color='green', marker='x')
    plt.grid(True)
    plt.show()