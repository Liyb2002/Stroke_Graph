import numpy as np
import torch


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def chosen_face_id(boundary_points, edge_index_face_edge_list, index_id, edge_features):

    if edge_features.shape[1] == 1:
        boundary_points_tensor = torch.stack(boundary_points).to(torch.float32)
        return boundary_points_tensor

    # pair the edges index with each face
    face_to_edges = {}
    for face_edge_pair in edge_index_face_edge_list:
        face_list_index = face_edge_pair[0]
        edge_list_index = face_edge_pair[1]

        face_id = index_id[0][face_list_index].item()
        edge_id = index_id[0][edge_list_index].item()

        if face_id not in face_to_edges:
            face_to_edges[face_id] = []
        face_to_edges[face_id].append(edge_id)

    # builds which points a face have
    face_to_points = {}
    for face_id, edge_ids in face_to_edges.items():
        unique_points = set()
        for edge_id in edge_ids:
            # get the points for the edge
            edge_points = edge_features[0, edge_id, :]

            start_point = edge_points[:3]
            end_point = edge_points[3:]
            
            # add each point to the set
            unique_points.add(start_point)
            unique_points.add(end_point)

        # store the unique points in the dictionary
        face_to_points[face_id] = list(unique_points)


    # Find which face has all its point in the boundary_point
    # output is the face_id
    target_face_id = 0
    num_faces = len(face_to_points)

    boundary_points_values_set = {tuple(torch.cat(boundary_point).to(torch.float32).tolist()) for boundary_point in boundary_points}

    for face_id, face_points in face_to_points.items():
        face_points_values_set = {tuple(face_point.tolist()) for face_point in face_points}
        if boundary_points_values_set.issubset(face_points_values_set):
            target_face_id = face_id
            break
    
    # Now, build the ground truth matrix
    gt_matrix = torch.zeros((num_faces, 1))
    gt_matrix[target_face_id, 0] = 1

    return gt_matrix


def find_left_edge(edge_features, node_features):
    # Extracting shapes
    _, n, _ = edge_features.shape
    _, m, _ = node_features.shape
    
    # Initialize the output tensor
    output = torch.ones((m, 1), dtype=torch.float32)
    
    # Iterate through each element in node_features
    for i in range(m):
        # Extract the current pair of 3D points from node_features
        node_pair = node_features[0, i, :]
        
        # Check if this pair is in any of the edge_features
        for j in range(n):
            if torch.equal(node_pair, edge_features[0, j, :]):
                output[i, 0] = 0
                break
    
    return output


def vis_gt_face(brep_edge_features, gt_index, edge_index_face_edge_list, index_id):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_edges = brep_edge_features.shape[1]

    face_to_edges = {}
    for face_edge_pair in edge_index_face_edge_list:
        face_list_index = face_edge_pair[0]
        edge_list_index = face_edge_pair[1]

        face_id = index_id[0][face_list_index].item()
        edge_id = index_id[0][edge_list_index].item()

        if face_id not in face_to_edges:
            face_to_edges[face_id] = []
        face_to_edges[face_id].append(edge_id)

    chosen_face = face_to_edges[gt_index]


    for i in range(num_edges):
        start_point = brep_edge_features[0, i, :3]
        end_point = brep_edge_features[0, i, 3:]

        col = 'blue'
        if i in chosen_face:
            col = 'red'

        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=col)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



def vis_predicted_face(brep_edge_features, predicted_index, edge_index_face_edge_list, index_id):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_edges = brep_edge_features.shape[1]

    face_to_edges = {}
    for face_edge_pair in edge_index_face_edge_list:
        face_list_index = face_edge_pair[0]
        edge_list_index = face_edge_pair[1]

        face_id = index_id[0][face_list_index].item()
        edge_id = index_id[0][edge_list_index].item()

        if face_id not in face_to_edges:
            face_to_edges[face_id] = []
        face_to_edges[face_id].append(edge_id)

    chosen_face = face_to_edges[predicted_index]


    for i in range(num_edges):
        start_point = brep_edge_features[0, i, :3]
        end_point = brep_edge_features[0, i, 3:]

        col = 'blue'
        if i in chosen_face:
            col = 'red'

        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=col)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



def vis_stroke_cloud(node_features):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    node_features = node_features.squeeze(0)

    # Plot all strokes in blue
    for stroke in node_features:
        start = stroke[:3].numpy()
        end = stroke[3:].numpy()
        
        # Plot the line segment for the stroke in blue
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], marker='o', color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



def vis_gt_strokes(brep_edge_features, gt_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_edges = brep_edge_features.shape[1]

    for i in range(num_edges):
        start_point = brep_edge_features[0, i, :3]
        end_point = brep_edge_features[0, i, 3:]

        col = 'blue'
        if gt_matrix[i] > 0.5:
            col = 'red'

        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=col)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



def vis_predicted_strokes(brep_edge_features, predicted_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    predicted_matrix_np = predicted_matrix.detach().numpy().flatten()
    max4indices = np.argsort(predicted_matrix_np)[-4:]
    print("np.argsort(predicted_matrix_np)", np.argsort(predicted_matrix_np))


    num_edges = brep_edge_features.shape[1]

    for i in range(num_edges):
        start_point = brep_edge_features[0, i, :3]
        end_point = brep_edge_features[0, i, 3:]

        col = 'blue'
        if i in max4indices:
            col = 'red'

        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=col)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



def chosen_edge_id(boundary_points, edge_features):
    # Convert boundary_points to a tensor
    boundary_points_tensor = torch.tensor(boundary_points, dtype=torch.float32)
    
    # Initialize the output matrix
    num_edges = edge_features.shape[1]
    gt_matrix = torch.zeros((num_edges, 1), dtype=torch.float32)
    
    # Loop through each edge
    for i in range(num_edges):
        start_point = edge_features[0, i, :3]
        end_point = edge_features[0, i, 3:]

        # Check if both the start and end points are in boundary_points
        start_in_boundary = any(torch.equal(start_point, bp) for bp in boundary_points_tensor)
        end_in_boundary = any(torch.equal(end_point, bp) for bp in boundary_points_tensor)
        
        # Set the value in gt_matrix
        if start_in_boundary and end_in_boundary:
            gt_matrix[i, 0] = 1
    
    return gt_matrix



def chosen_vertex_id(boundary_points, vertex_features):
    # Convert boundary_points to a tensor
    boundary_points_tensor = torch.tensor(boundary_points, dtype=torch.float32)
    
    # Initialize the output matrix
    num_verts = vertex_features.shape[0]
    gt_matrix = torch.zeros((num_verts), dtype=torch.float32)
    
    # Loop through each edge
    for i in range(num_verts):
        start_point = vertex_features[i, :3]

        # Check if both the start and end points are in boundary_points
        start_in_boundary = any(torch.equal(start_point, bp) for bp in boundary_points_tensor)
        
        # Set the value in gt_matrix
        if start_in_boundary:
            gt_matrix[i] = 1
    
    # print("gt_matrix", gt_matrix)
    return gt_matrix


def chosen_edge_id_stroke_cloud(boundary_points, node_features):

    boundary_points_list = [[float(point[0]), float(point[1]), float(point[2])] for point in boundary_points]
    print("boundary_points", boundary_points_list)
    
    # Initialize the output matrix
    num_edges = node_features.shape[1]
    gt_matrix = torch.zeros((num_edges, 1), dtype=torch.float32)
    
    # Loop through each edge
    for i in range(num_edges):
        start_point = node_features[0, i, :3]  # Assuming x, y, z are the first three dimensions
        end_point = node_features[0, i, 3:]    # Assuming x, y, z are the next three dimensions
        
        print("start_point", start_point, "end_point", end_point)
        # Check if start and end points are on the plane (exist in boundary_points)
        start_on_plane = (start_point[0] in boundary_points[0]) and (start_point[1] in boundary_points[1]) and (start_point[2] in boundary_points[2])
        end_on_plane = (end_point[0] in boundary_points[0]) and (end_point[1] in boundary_points[1]) and (end_point[2] in boundary_points[2])
        
        if start_on_plane and end_on_plane:
            gt_matrix[i] = 1.0  # Mark the edge as on the plane
        
    return gt_matrix



def edit_stroke_cloud(chosen_edges, node_features, operations_matrix, intersection_matrix, operations_order_matrix):
    # Convert chosen_edges to boolean mask based on >0.5 threshold
    chosen_mask = chosen_edges > 0.5
    chosen_mask = chosen_mask.squeeze(0)  # Remove the first dimension if present
    
    # Select indices of chosen edges
    chosen_indices = torch.nonzero(chosen_mask, as_tuple=False)[:, 0] 
    
    # Extract chosen node features
    node_features = node_features[:, chosen_indices, :]

    # Extract corresponding operations matrix
    operations_matrix = operations_matrix[:, chosen_indices, :]
    
    # Extract intersection matrix for chosen edges
    intersection_matrix = intersection_matrix[:, chosen_indices][:, :, chosen_indices]
    
    # Extract operations order matrix for chosen edges
    operations_order_matrix = operations_order_matrix[:, chosen_indices, :]
    
    return node_features, operations_matrix, intersection_matrix, operations_order_matrix
