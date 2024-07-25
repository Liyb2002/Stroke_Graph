import numpy as np
from itertools import permutations, combinations
import torch
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def vis_planar(plane_chosen, plane_to_node, node_features):
    node_features = node_features.squeeze(0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_planes = plane_chosen.shape[0]
    num_edges = node_features.shape[0]

    for i in range(num_edges):
        start_point = node_features[i, :3]
        end_point = node_features[i, 3:]


        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color='blue')


    for i in range(num_planes):
        if plane_chosen[i] > 0.3:
            chosen_indices = plane_to_node[i]
            for idx in chosen_indices:
                start_point = node_features[idx, :3]
                end_point = node_features[idx, 3:]

                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    


def face_aggregate_withMask(stroke_matrix, mask, min_threshold = 0.2):

    threshold = 0.5
    selected_indices = torch.zeros((stroke_matrix.shape[0], 1), dtype=torch.float32)

    while True:
        if threshold < min_threshold:
            return selected_indices

        chosen_indices = (mask.squeeze() > threshold).nonzero(as_tuple=True)[0]
        group = satisfy(chosen_indices, stroke_matrix)

        if len(group) == 0:
            threshold -= 0.1
            continue
        
        selected_indices[torch.tensor(group)] = 1
        return selected_indices


    


def satisfy(chosen_indices, stroke_matrix):
    def all_points_repeated_twice(strokes):
        points = strokes.reshape(-1, 3)
        unique_points, counts = torch.unique(points, return_counts=True, dim=0)
        return torch.all(counts == 2)

    possible_groups = list(itertools.chain(
        itertools.combinations(chosen_indices.tolist(), 3),
        itertools.combinations(chosen_indices.tolist(), 4)
    ))

    for group in possible_groups:
        strokes = stroke_matrix[list(group)]
        if all_points_repeated_twice(strokes):
            return group

    return []



def sketch_to_normal(sketch):

    sketch_tensor = torch.tensor(sketch, dtype=torch.float32)
    
    # Extract all the points from the sketch
    points = sketch_tensor.view(-1, 3)
    
    # Check for coplanarity by finding the axis with the same value for all points
    unique_x = torch.unique(points[:, 0])
    unique_y = torch.unique(points[:, 1])
    unique_z = torch.unique(points[:, 2])

    if unique_x.size(0) == 1:
        return [1, 0, 0]
    elif unique_y.size(0) == 1:
        return [0, 1, 0]
    
    return [0, 0, 1]


def extract_unique_points(sketch):
    # Convert sketch to a list of tuples for easier manipulation
    strokes = [((stroke[:3].tolist(), stroke[3:].tolist())) for stroke in sketch]

    # Start with the first stroke
    current_stroke = strokes.pop(0)
    points_list = [current_stroke[1]]
    # Find the next stroke and continue until a loop is formed
    while strokes:
        end_point = points_list[-1]
        found = False
        for i, (start_point, next_point) in enumerate(strokes):
            if start_point == end_point:
                points_list.append(next_point)
                strokes.pop(i)
                found = True
                break
            elif next_point == end_point:
                points_list.append(start_point)
                strokes.pop(i)
                found = True
                break
        if not found:
            raise ValueError("Cannot find a continuous path with the given strokes")

    # Convert the list of points to a numpy array
    unique_points_np = np.array(points_list)

    return unique_points_np



def get_extrude_amount(stroke_features, chosen_matrix, sketch_strokes):
    # Ensure chosen_matrix and sketch_strokes are boolean tensors for indexing
    chosen_mask = chosen_matrix > 0.5
    sketch_mask = sketch_strokes > 0.5
    
    # Use the mask to index into stroke_features and get the chosen strokes
    chosen_stroke_features = stroke_features[chosen_mask.squeeze()]
    
    # Find any of the sketch strokes
    sketch_stroke_features = stroke_features[sketch_mask.squeeze()]
    sketch_point = sketch_stroke_features[0][:3]  # Use the first point of the first sketch stroke
    
    distances = []
    direction = None
    
    # Calculate the distance for each chosen stroke and determine the direction
    for stroke in chosen_stroke_features:
        point1 = stroke[:3]
        point2 = stroke[3:]
        
        # Determine the direction if not already determined
        if direction is None and (torch.all(point1 == sketch_point) or torch.all(point2 == sketch_point)):
            if torch.all(point1 == sketch_point):
                direction = point2 - point1
            else:
                direction = point1 - point2
        
        # Calculate the absolute difference between the points
        diff = torch.abs(point1 - point2)
        
        # Find the maximum difference which corresponds to the distance
        distance = torch.max(diff)
        distances.append(distance.item())
    
    # Find the maximum distance
    max_distance = max(distances) if distances else 0
    
    # Normalize the direction
    if direction is not None:
        direction = direction / torch.norm(direction)
    
    return max_distance, direction


def brep_all_covered(brep_features, stroke_features):
    print("stroke_features", stroke_features)
    stroke_set = {tuple(round(val, 3) for val in row.tolist()) for row in stroke_features}

    # Check if all brep features are covered by stroke features
    for brep in brep_features:
        if tuple(round(val, 3) for val in brep[1]) not in stroke_set:
            print(f"Feature not contained: {brep}")

