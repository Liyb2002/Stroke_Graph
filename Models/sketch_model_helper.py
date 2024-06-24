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

        face_id = index_id[face_list_index].item()
        edge_id = index_id[edge_list_index].item()

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



def vis_gt_face(brep_edge_features, gt_index, edge_index_face_edge_list, index_id):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_edges = brep_edge_features.shape[1]

    face_to_edges = {}
    for face_edge_pair in edge_index_face_edge_list:
        face_list_index = face_edge_pair[0]
        edge_list_index = face_edge_pair[1]

        face_id = index_id[face_list_index].item()
        edge_id = index_id[edge_list_index].item()

        if face_id not in face_to_edges:
            face_to_edges[face_id] = []
        face_to_edges[face_id].append(edge_id)

    chosen_face = face_to_edges[gt_index.item()]


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

        face_id = index_id[face_list_index].item()
        edge_id = index_id[edge_list_index].item()

        if face_id not in face_to_edges:
            face_to_edges[face_id] = []
        face_to_edges[face_id].append(edge_id)

    chosen_face = face_to_edges[predicted_index[0]]


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

