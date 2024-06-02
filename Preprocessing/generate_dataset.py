import proc_CAD.proc_gen
import proc_CAD.CAD_to_stroke_cloud
import proc_CAD.render_images
import proc_CAD.Program_to_STL
import proc_CAD.helper

import gnn_graph
import SBGCN.run_SBGCN
import SBGCN.brep_read

import shutil
import os
import pickle


class dataset_generator():

    def __init__(self, number_data = 20):
        self.number_data = number_data
        self.SBGCN_encoder = SBGCN.run_SBGCN.load_pretrained_SBGCN_model()
        self.generate_dataset()
 

    def generate_dataset(self):
        successful_generations = 0
        # Create directory /dataset/ if it doesn't exist
        if os.path.exists('dataset'):
            shutil.rmtree('dataset')
        os.makedirs('dataset', exist_ok=True)
        
        while successful_generations < self.number_data:
            if self.generate_single_data(successful_generations):
                successful_generations += 1
            else:
                print("Retrying...")

    def generate_single_data(self, successful_generations):
        # Create directory /dataset/data_{successful_generations}
        data_directory = f'dataset/data_{successful_generations}'
        os.makedirs(data_directory, exist_ok=True)
        
        # Generate a new program & save the brep
        try:
            # Pass in the directory to the simple_gen function
            proc_CAD.proc_gen.random_program(data_directory)
            # proc_CAD.proc_gen.simple_gen(data_directory)

            # Create brep for the new program and pass in the directory
            valid_parse = proc_CAD.Program_to_STL.run(data_directory)
        except Exception as e:
            print(f"An error occurred: {e}")
            shutil.rmtree(data_directory)
            return False
        
        if not valid_parse:
            shutil.rmtree(data_directory)
            return False
        
        
        # Save matrices for stroke_cloud_graph
        stroke_cloud= proc_CAD.CAD_to_stroke_cloud.run(data_directory)
        node_features, operations_matrix, intersection_matrix = gnn_graph.build_graph(stroke_cloud)
        stroke_cloud_save_path = os.path.join(data_directory, 'stroke_cloud_graph.pkl')

        with open(stroke_cloud_save_path, 'wb') as f:
            pickle.dump({
                'node_features': node_features,
                'operations_matrix': operations_matrix,
                'intersection_matrix': intersection_matrix
            }, f)


        # Save matrices for Brep Embedding
        brep_directory = os.path.join(data_directory, 'canvas')
        for file_name in os.listdir(brep_directory):
            if file_name.startswith('brep_') and file_name.endswith('.step'):
                brep_file_path = os.path.join(brep_directory, file_name)
                brep_graph = SBGCN.brep_read.create_graph_from_step_file(brep_file_path)
                face_embeddings, edge_embeddings, vertex_embeddings = self.SBGCN_encoder.embed(brep_graph)

                # extract index i
                index = file_name.split('_')[1].split('.')[0]
                os.makedirs(os.path.join(data_directory, 'embedding'), exist_ok=True)
                embeddings_file_path = os.path.join(data_directory, 'embedding', f'embedding_{index}.pkl')
                with open(embeddings_file_path, 'wb') as f:
                    pickle.dump({
                        'face_embeddings': face_embeddings,
                        'edge_embeddings': edge_embeddings,
                        'vertex_embeddings': vertex_embeddings
                    }, f)


        return True


def run():
    d_generator = dataset_generator()
    d_generator.generate_dataset()

run()