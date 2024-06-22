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



    for face_id, points in face_to_points.items():
        print(f"Face ID: {face_id}")
        print("Points:")
        for point in points:
            print(point)

