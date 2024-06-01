import proc_CAD.proc_gen
import proc_CAD.CAD_to_stroke_cloud
import proc_CAD.render_images
import proc_CAD.Program_to_STL
import proc_CAD.helper

import gnn_graph
import SBGCN.run_SBGCN

import os
import pickle

def generate_dataset(number_data=20):
    successful_generations = 0
    # Create directory /dataset/ if it doesn't exist
    os.makedirs('dataset', exist_ok=True)
    
    while successful_generations < number_data:
        if generate_single_data(successful_generations):
            successful_generations += 1
        else:
            print("Retrying...")

def generate_single_data(successful_generations):
    # Create directory /dataset/data_{successful_generations}
    data_directory = f'dataset/data_{successful_generations}'
    os.makedirs(data_directory, exist_ok=True)
    
    try:
        # Pass in the directory to the simple_gen function
        proc_CAD.proc_gen.random_program(data_directory)

        # Create brep for the new program and pass in the directory
        proc_CAD.Program_to_STL.run(data_directory)
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

    stroke_cloud= proc_CAD.CAD_to_stroke_cloud.run(data_directory)
    stroke_cloud_graph = gnn_graph.build_graph(stroke_cloud)
    stroke_cloud_save_path = os.path.join(data_directory, 'stroke_cloud_graph.pkl')

    with open(stroke_cloud_save_path, 'wb') as f:
        pickle.dump(stroke_cloud_graph, f)

    return True


def run():
    generate_dataset()

run()