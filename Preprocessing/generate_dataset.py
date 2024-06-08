import Preprocessing.proc_CAD.proc_gen
import Preprocessing.proc_CAD.CAD_to_stroke_cloud
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

class dataset_generator():

    def __init__(self):
        self.SBGCN_encoder = Preprocessing.SBGCN.run_SBGCN.load_pretrained_SBGCN_model()
        # if os.path.exists('dataset'):
        #     shutil.rmtree('dataset')
        # os.makedirs('dataset', exist_ok=True)

        self.generate_dataset('dataset/train_dataset', number_data = 2000, start = 2000)
        self.generate_dataset('dataset/eval_dataset', number_data = 200, start = 200)
 

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
        stroke_cloud= Preprocessing.proc_CAD.CAD_to_stroke_cloud.run(data_directory)
        node_features, operations_matrix, intersection_matrix, operations_order_matrix= Preprocessing.gnn_graph.build_graph(stroke_cloud)
        stroke_cloud_save_path = os.path.join(data_directory, 'stroke_cloud_graph.pkl')

        with open(stroke_cloud_save_path, 'wb') as f:
            pickle.dump({
                'node_features': node_features,
                'operations_matrix': operations_matrix,
                'intersection_matrix': intersection_matrix,
                'operations_order_matrix': operations_order_matrix
            }, f)


        # 3) Save matrices for Brep Embedding
        brep_directory = os.path.join(data_directory, 'canvas')
        for file_name in os.listdir(brep_directory):
            if file_name.startswith('brep_') and file_name.endswith('.step'):
                brep_file_path = os.path.join(brep_directory, file_name)
                face_features_list, edge_features_list, vertex_features_list, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id= Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)

                
                face_features = Preprocessing.proc_CAD.helper.preprocess_features(face_features_list)
                edge_features = Preprocessing.proc_CAD.helper.preprocess_features(edge_features_list)
                vertex_features = Preprocessing.proc_CAD.helper.preprocess_features(vertex_features_list)

                # extract index i
                index = file_name.split('_')[1].split('.')[0]
                os.makedirs(os.path.join(data_directory, 'brep_embedding'), exist_ok=True)
                embeddings_file_path = os.path.join(data_directory, 'brep_embedding', f'brep_info_{index}.pkl')
                with open(embeddings_file_path, 'wb') as f:
                    pickle.dump({
                        'face_features': face_features,
                        'edge_features': edge_features,
                        'vertex_features': vertex_features,
                        
                        'edge_index_face_edge_list': edge_index_face_edge_list,
                        'edge_index_edge_vertex_list': edge_index_edge_vertex_list,
                        'edge_index_face_face_list': edge_index_face_face_list,

                        'index_id': torch.tensor(index_id, dtype=torch.int64)

                    }, f)

        # 4) Save rendered 2D image
        Preprocessing.proc_CAD.render_images.run_render_images(data_directory)


        return True

