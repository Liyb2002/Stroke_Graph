import numpy as np
from itertools import permutations, combinations

def face_aggregate(stroke_matrix):
    """
    This function permutes all the strokes and groups them into groups of 3 or 4.

    Parameters:
    stroke_matrix (numpy.ndarray): A matrix of shape (num_strokes, 6) where each row represents a stroke
                                   with start and end points in 3D space.

    Returns:
    list: A list of groups of strokes, where each group contains either 3 or 4 strokes.
    """
    
    # Ensure input is a numpy array
    stroke_matrix = np.array(stroke_matrix)
    
    # Reshape the stroke matrix to remove the leading dimension of 1
    stroke_matrix = stroke_matrix.reshape(-1, 6)
    
    # Get the number of strokes
    num_strokes = stroke_matrix.shape[0]
    
    # Generate all combinations of groups of 3 and 4 strokes
    groups_of_3 = list(combinations(stroke_matrix, 3))
    groups_of_4 = list(combinations(stroke_matrix, 4))
    
    # Combine the groups into a single list
    all_groups = groups_of_3 + groups_of_4
    
    def are_strokes_coplanar(group):
        for dim in range(3):  # Check each of x, y, z
            start_points = group[:, dim]
            end_points = group[:, dim + 3]
            if np.all(start_points == start_points[0]) and np.all(end_points == end_points[0]):
                return True
        return False
    
    def are_strokes_connected(group):
        points = np.concatenate([group[:, :3], group[:, 3:]], axis=0)
        unique_points, counts = np.unique(points, axis=0, return_counts=True)
        return np.all(counts == 2)

    # Filter out groups that are not coplanar
    coplanar_groups = [group for group in all_groups if are_strokes_coplanar(np.array(group))]
    valid_groups = [group for group in coplanar_groups if are_strokes_connected(np.array(group))]

    return valid_groups
