import Preprocessing.proc_CAD.proc_gen
import Preprocessing.proc_CAD.CAD_to_stroke_cloud
import Preprocessing.proc_CAD.draw_all_lines

import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.helper
import Preprocessing.proc_CAD.render_images

import Preprocessing.gnn_graph
import Preprocessing.SBGCN.run_SBGCN
import Preprocessing.SBGCN.brep_read

import shutil
import os
import pickle
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

class dataset_generator():

    def __init__(self):
        # if os.path.exists('dataset'):
        #     shutil.rmtree('dataset')
        # os.makedirs('dataset', exist_ok=True)

        self.generate_dataset('dataset/extrude_only_simple', number_data = 0, start = 0)
        self.generate_dataset('dataset/extrude_only_test', number_data = 0, start = 0)
        self.generate_dataset('dataset/extrude_only_eval', number_data = 0, start = 0)
        self.generate_dataset('dataset/drawing', number_data = 1, start = 0)


    def generate_dataset(self, dir, number_data, start):
        successful_generations = start

        while successful_generations < number_data:
            if self.generate_single_data(successful_generations, dir):
                successful_generations += 1
            else:
                print("Retrying...")

    def generate_single_data(self, successful_generations, dir):
        data_directory = os.path.join(dir, f'data_{successful_generations}')
        if os.path.exists(data_directory):
            shutil.rmtree(data_directory)

        os.makedirs(data_directory, exist_ok=True)
        
        # Generate a new program & save the brep
        try:
            # Pass in the directory to the simple_gen function
            Preprocessing.proc_CAD.proc_gen.random_program(data_directory)
            # Preprocessing.proc_CAD.proc_gen.simple_gen(data_directory)

            # Create brep for the new program and pass in the directory
            valid_parse = Preprocessing.proc_CAD.Program_to_STL.run(data_directory)
        except Exception as e:
            print(f"An error occurred: {e}")
            shutil.rmtree(data_directory)
            return False
        
        if not valid_parse:
            shutil.rmtree(data_directory)
            return False
        
        
        # 1) Save matrices for stroke_cloud_graph
        stroke_cloud_edges, stroke_cloud_faces= Preprocessing.proc_CAD.draw_all_lines.run(data_directory)
        node_features, operations_matrix, intersection_matrix, operations_order_matrix= Preprocessing.gnn_graph.build_graph(stroke_cloud_edges)
        stroke_cloud_save_path = os.path.join(data_directory, 'stroke_cloud_graph.pkl')

        face_to_stroke = Preprocessing.proc_CAD.helper.face_to_stroke(stroke_cloud_faces, node_features)
        with open(stroke_cloud_save_path, 'wb') as f:
            pickle.dump({
                'node_features': node_features,
                'operations_matrix': operations_matrix,
                'intersection_matrix': intersection_matrix,
                'operations_order_matrix': operations_order_matrix,
                'face_aggregate': face_to_stroke
            }, f)


        # 3) Save matrices for Brep Embedding
        brep_directory = os.path.join(data_directory, 'canvas')
        brep_files = [file_name for file_name in os.listdir(brep_directory)
              if file_name.startswith('brep_') and file_name.endswith('.step')]
        brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        for i in range(1, len(brep_files) + 1):
            sublist = brep_files[:i]
            final_brep_edges = []
            final_brep_coplanar = []

            prev_brep_edges = []

            # Run the cascade brep features algorithm
            for file_name in sublist:
                brep_file_path = os.path.join(brep_directory, file_name)
                
                # len(edge_coplanar_list) = num_faces
                edge_features_list, edge_coplanar_list= Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)

                # If this is the first brep
                if len(prev_brep_edges) == 0:
                    final_brep_edges = edge_features_list
                    prev_brep_edges = edge_features_list
                    new_features = edge_features_list
                    final_brep_coplanar = edge_coplanar_list
                else:
                    # We already have brep
                    new_features, new_planes= find_new_features(prev_brep_edges, edge_features_list, edge_coplanar_list) 
                    final_brep_edges += new_features
                    final_brep_coplanar += new_planes
                    prev_brep_edges = edge_features_list

            # Now write the brep features
            
            index = sublist[-1].split('_')[1].split('.')[0]
            os.makedirs(os.path.join(data_directory, 'brep_embedding'), exist_ok=True)
            embeddings_file_path = os.path.join(data_directory, 'brep_embedding', f'brep_info_{index}.pkl')
            with open(embeddings_file_path, 'wb') as f:
                pickle.dump({
                    'final_brep_edges': final_brep_edges,
                    'final_brep_coplanar': final_brep_coplanar, 
                    'new_features': new_features
                }, f)


        # 4) Save rendered 2D image
        # Preprocessing.proc_CAD.render_images.run_render_images(data_directory)


        return True



def find_new_features(final_brep_edges, edge_features_list, edge_coplanar_list):
    def is_same_direction(line1, line2):
        """Check if two lines have the same direction."""
        vector1 = np.array(line1[3:]) - np.array(line1[:3])
        vector2 = np.array(line2[3:]) - np.array(line2[:3])
        return np.allclose(vector1 / np.linalg.norm(vector1), vector2 / np.linalg.norm(vector2))

    def is_point_on_line(point, line):
        """Check if a point lies on a given line."""
        start, end = np.array(line[:3]), np.array(line[3:])
        return np.allclose(np.cross(end - start, point - start), 0)

    def is_line_contained(line1, line2):
        """Check if line1 is contained within line2."""
        return is_point_on_line(np.array(line1[:3]), line2) and is_point_on_line(np.array(line1[3:]), line2)

    def replace_line_in_faces(faces, old_line, new_line):
        """Replace the old line with the new line in all faces."""
        for face in faces:
            for i in range(len(face)):
                if np.allclose(face[i], old_line):
                    face[i] = new_line

    new_features = []

    for edge_line in edge_features_list:
        relation_found = False

        for brep_line in final_brep_edges:
            if np.allclose(edge_line, brep_line):
                # Relation 1: The two lines are exactly the same
                relation_found = True
                break
            
            elif is_same_direction(edge_line, brep_line) and is_line_contained(brep_line, edge_line):
                # Relation 2: edge_features_list line contains final_brep_edges line
                relation_found = True
                
                if np.allclose(edge_line[:3], brep_line[:3]) or np.allclose(edge_line[:3], brep_line[3:]):
                    new_line = brep_line[3:] + edge_line[3:]
                else:
                    new_line = brep_line[:3] + edge_line[3:]

                replace_line_in_faces(edge_coplanar_list, brep_line, new_line)
                new_features.append(new_line)
                break
            
            elif is_same_direction(edge_line, brep_line) and is_line_contained(edge_line, brep_line):
                # Relation 3: final_brep_edges line contains edge_features_list line
                relation_found = True
                
                if np.allclose(edge_line[:3], brep_line[:3]) or np.allclose(edge_line[:3], brep_line[3:]):
                    new_line = edge_line[3:] + brep_line[3:]
                else:
                    new_line = edge_line[:3] + brep_line[3:]

                replace_line_in_faces(edge_coplanar_list, brep_line, new_line)
                new_features.append(new_line)
                break
        
        if not relation_found:
            # Relation 4: None of the relations apply
            new_features.append(edge_line)

    new_planes = []
    for face in edge_coplanar_list:
        for line in face:
            if any(np.allclose(line, new_feature) for new_feature in new_features):
                new_planes.append(face)
                break

    return new_features, new_planes





def vis_compare(node_features, edge_features):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot node_features in blue and compute limits
    for stroke in node_features:
        start = stroke[:3].numpy()  # Start point of the stroke (x1, y1, z1)
        end = stroke[3:].numpy()    # End point of the stroke (x2, y2, z2)
        
        # Update the min and max limits for each axis
        x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
        y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
        z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

        # Plot the line segment for the stroke in blue
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], marker='o', color='blue')

    # Plot edge_features in green and update limits
    for edge in edge_features:
        start = edge[:3].numpy()  # Start point of the edge (x1, y1, z1)
        end = edge[3:].numpy()    # End point of the edge (x2, y2, z2)
        
        # Update the min and max limits for each axis
        x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
        y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
        z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

        # Plot the line segment for the edge in green
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], marker='o', color='green')

    # Compute the center of the shape
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    # Compute the maximum difference across x, y, z directions
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Set the same limits for x, y, and z axes centered around the computed center
    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
