import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
import math

NUM_POINTS_LEFT = 45
NUM_POINTS_RIGHT = 30
RESAMPLE_SPACING = 0.2  # meters
OPTIMIZATION_ITERATIONS = 500
LEARNING_RATE = 0.1
MARGIN = 0.3  # meters (safety margin from boundaries)

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

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

def fit_spline(points):
    """Fit cubic spline and resample at constant spacing"""
    points = np.array(points)
    
    # Handle case with too few points
    if len(points) < 4:
        # Add extra points to meet spline requirements
        points = np.vstack([points, points[-1:]])
    
    # Compute cumulative distance along the path
    if len(points) > 1:
        dist = np.zeros(len(points))
        dist[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        dist = dist/dist[-1]
    else:
        dist = np.linspace(0, 1, len(points))
    
    # Fit spline
    tck, u = splprep(points.T, u=dist, s=0)
    
    # Resample at constant spacing along the curve
    u_new = np.linspace(0, 1, int(dist[-1]/RESAMPLE_SPACING))
    dense_points = np.array(splev(u_new, tck)).T
    tangents = np.array(splev(u_new, tck, der=1)).T
    
    # Normalize tangents
    tangent_norms = np.linalg.norm(tangents, axis=1)
    tangents = tangents / tangent_norms[:, np.newaxis]
    
    return dense_points, tangents

def compute_normals(tangents):
    """Compute normal vectors from tangents (90 degree rotation)"""
    return np.array([[-t[1], t[0]] for t in tangents])

def compute_boundary_offsets(centerline, normals, left_cones, right_cones):
    """Compute boundary offsets for each centerline point"""
    left_tree = KDTree(left_cones)
    right_tree = KDTree(right_cones)
    
    left_offsets = []
    right_offsets = []
    
    for i, (pt, normal) in enumerate(zip(centerline, normals)):
        # Find closest left boundary point
        _, lidx = left_tree.query(pt)
        lvec = left_cones[lidx] - pt
        left_offset = np.dot(lvec, normal)
        
        # Find closest right boundary point
        _, ridx = right_tree.query(pt)
        rvec = right_cones[ridx] - pt
        right_offset = np.dot(rvec, normal)
        
        # Apply safety margin and store
        left_offsets.append(left_offset - MARGIN)
        right_offsets.append(right_offset + MARGIN)
    
    return np.array(left_offsets), np.array(right_offsets)

def compute_gradient(racing_line):
    """Compute gradient of curvature squared using finite differences"""
    epsilon = 1e-6
    grad = np.zeros_like(racing_line)
    base_curvature = total_curvature_squared(racing_line)
    
    for i in range(1, len(racing_line)-1):  # Skip endpoints
        for j in range(2):  # x and y dimensions
            # Perturb point
            original = racing_line[i, j]
            racing_line[i, j] = original + epsilon
            
            # Compute new curvature
            new_curvature = total_curvature_squared(racing_line)
            
            # Finite difference
            grad[i, j] = (new_curvature - base_curvature) / epsilon
            racing_line[i, j] = original
            
    return grad

def total_curvature_squared(points):
    """Compute sum of squared curvature for path"""
    # Compute vectors between points
    v1 = points[1:-1] - points[:-2]   # Backward vectors
    v2 = points[2:] - points[1:-1]     # Forward vectors
    
    # Compute curvature using cross product method
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    
    # Handle zero-length vectors
    valid = (norm_v1 > 1e-6) & (norm_v2 > 1e-6)
    curvature = np.zeros(len(cross))
    curvature[valid] = cross[valid] / (norm_v1[valid] * norm_v2[valid])
    
    return np.sum(curvature**2)

def enforce_boundary_constraints(racing_line, centerline, normals, 
                                left_offsets, right_offsets, centerline_tree):
    """Project points to track boundaries using Frenet frame"""
    for i, point in enumerate(racing_line):
        # Find nearest centerline point
        _, idx = centerline_tree.query(point)
        c_point = centerline[idx]
        normal = normals[idx]
        
        # Compute lateral offset
        offset = np.dot(point - c_point, normal)
        
        # Apply boundary constraints
        if offset < left_offsets[idx]:
            offset = left_offsets[idx]
        elif offset > right_offsets[idx]:
            offset = right_offsets[idx]
        
        # Project point to constrained position
        racing_line[i] = c_point + offset * normal
        
    return racing_line

def optimize_raceline(mid_x, mid_y, left_cones, right_cones):
    """Transform midline into optimized racing line"""
    # Convert midline to point array
    centerline_initial = np.column_stack([mid_x, mid_y])
    
    # Create dense centerline with spline
    dense_centerline, tangent_vectors = fit_spline(centerline_initial)
    
    # Compute normals and boundary offsets
    centerline_normals = compute_normals(tangent_vectors)
    left_offsets, right_offsets = compute_boundary_offsets(
        dense_centerline, centerline_normals, left_cones, right_cones
    )
    
    # Build KDTree for centerline projection
    centerline_tree = KDTree(dense_centerline)
    
    # Initialize racing line at centerline
    racing_line = dense_centerline.copy()
    
    # Optimize racing line
    for _ in range(OPTIMIZATION_ITERATIONS):
        # Compute gradient (finite difference)
        gradient = compute_gradient(racing_line)
        
        # Update racing line position
        racing_line -= LEARNING_RATE * gradient
        
        # Project points to track boundaries
        racing_line = enforce_boundary_constraints(
            racing_line, dense_centerline, centerline_normals,
            left_offsets, right_offsets, centerline_tree
        )
    
    return racing_line, dense_centerline

#-----------------------------

if __name__ == "__main__":
    t_vals_left = np.linspace(0, 2 * np.pi, NUM_POINTS_LEFT)
    t_vals_right = np.linspace(0, 2 * np.pi, NUM_POINTS_RIGHT)
    
    left_x = 50 * np.cos(t_vals_left) + 50
    left_y = 30 * np.sin(t_vals_left) + 30
    right_x = 45 * np.cos(t_vals_right) + 50
    right_y = 25 * np.sin(t_vals_right) + 30

    yellow_cones = np.column_stack((left_x, left_y))
    blue_cones = np.column_stack((right_x, right_y))

    mid_x, mid_y = find_midline(left_x, left_y, right_x, right_y)

    racing_line, dense_centerline = optimize_raceline(mid_x, mid_y, yellow_cones, blue_cones)
    
    plt.figure(figsize=(10,6))
    plt.plot(left_x, left_y, color='blue', marker='o')
    plt.plot(right_x, right_y, color='gold', marker='o')
    plt.plot(mid_x, mid_y, color='green', marker='x')
    plt.plot(racing_line[:, 0], racing_line[:, 1], color='red')
    print(racing_line)
    plt.grid(True)
    plt.show()