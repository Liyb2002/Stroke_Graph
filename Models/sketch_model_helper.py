import numpy as np
import torch


def chosen_face_id(boundary_points, edge_index_face_edge_list, index_id, edge_features):

    if edge_features.shape[1] == 1:
        return

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
    boundary_points_values_set = {tuple(torch.cat(boundary_point).to(torch.float32).tolist()) for boundary_point in boundary_points}

    for face_id, face_points in face_to_points.items():
        face_points_values_set = {tuple(face_point.tolist()) for face_point in face_points}
        if boundary_points_values_set.issubset(face_points_values_set):
            print("face_id!", face_id)
            break



    