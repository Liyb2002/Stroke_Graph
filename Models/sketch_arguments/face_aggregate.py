import numpy as np
from itertools import permutations, combinations
import torch

def face_aggregate(stroke_matrix):
    """
    This function permutes all the strokes and groups them into groups of 3 or 4.
    It then checks if all the strokes in each group are coplanar and connected.

    Parameters:
    stroke_matrix (numpy.ndarray): A matrix of shape (1, num_strokes, 6) where each row represents a stroke
                                   with start and end points in 3D space.

    Returns:
    list: A list of indices of coplanar and connected groups of strokes, where each group contains either 3 or 4 strokes.
    """
    
    # Ensure input is a numpy array
    stroke_matrix = np.array(stroke_matrix)
    
    # Remove the leading dimension of 1
    stroke_matrix = stroke_matrix.reshape(-1, 6)
    
    # Get the number of strokes
    num_strokes = stroke_matrix.shape[0]
    
    # Generate all combinations of indices for groups of 3 and 4 strokes
    indices = np.arange(num_strokes)
    groups_of_3 = list(combinations(indices, 3))
    groups_of_4 = list(combinations(indices, 4))
    
    # Combine the groups into a single list
    all_groups = groups_of_3 + groups_of_4
    
    # Function to check if a group of strokes are coplanar
    def are_strokes_coplanar(group_indices):
        group = stroke_matrix[np.array(group_indices)]
        for dim in range(3):  # Check each of x, y, z
            start_points = group[:, dim]
            end_points = group[:, dim + 3]
            if np.all(start_points == start_points[0]) and np.all(end_points == end_points[0]):
                return True
        return False
    
    # Function to check if a group of strokes are connected
    def are_strokes_connected(group_indices):
        group = stroke_matrix[np.array(group_indices)]
        points = np.concatenate([group[:, :3], group[:, 3:]], axis=0)
        unique_points, counts = np.unique(points, axis=0, return_counts=True)
        return np.all(counts == 2)
    
    # First filter the groups to get coplanar groups
    coplanar_groups = [group for group in all_groups if are_strokes_coplanar(group)]

    # Then filter the coplanar groups to get the connected groups
    valid_groups = [group for group in coplanar_groups if are_strokes_connected(group)]
    
    return valid_groups


def get_kth_operation(op_to_index_matrix, k):    
    squeezed_matrix = op_to_index_matrix.squeeze(0)
    kth_operation = squeezed_matrix[:, k].unsqueeze(1)

    return kth_operation


def build_gt_matrix(kth_operation, planes):
    num_planes = len(planes)
    gt_matrix = torch.zeros((num_planes, 1), dtype=torch.float32)

    # chosen_strokes = [i for i, val in enumerate(kth_operation) if val == 1]
    # print("Strokes with value 1 in kth_operation:", chosen_strokes)

    for idx, plane in enumerate(planes):
        chosen_count = sum(kth_operation[stroke] == 1 for stroke in plane)
        if chosen_count >= 3:
            gt_matrix[idx] = 1.0

    return gt_matrix
