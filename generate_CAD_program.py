import Preprocessing.dataloader
import Preprocessing.gnn_graph_Gen
import Preprocessing.gnn_graph_full
import Preprocessing.generate_dataset

import Preprocessing.proc_CAD.generate_program
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.brep_read
import Encoders.gnn_full.gnn

import Models.sketch_arguments.face_aggregate
import Models.sketch_model_helper

from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil

import random

# --------------------- Dataset --------------------- #
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/extrude_only_eval')
good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 0]
filtered_dataset = Subset(dataset, good_data_indices)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


# --------------------- Directory --------------------- #
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, 'program_output')
output_relative_dir = ('program_output/canvas')

# --------------------- Networks --------------------- #

# Operation Prediction
Op_dir = os.path.join(current_dir, 'checkpoints', 'operation_prediction')
Op_graph_encoder = Encoders.gnn_full.gnn.SemanticModule()
Op_graph_decoder = Encoders.gnn_full.gnn.Program_prediction()
Op_graph_encoder.load_state_dict(torch.load(os.path.join(Op_dir, 'graph_encoder.pth')))
Op_graph_decoder.load_state_dict(torch.load(os.path.join(Op_dir, 'graph_decoder.pth')))

def Op_predict(gnn_graph, current_program):
    x_dict = Op_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    output = Op_graph_decoder(x_dict, current_program)
    _, predicted_class = torch.max(output, 0)
    return predicted_class


# Sketch Prediction
sketch_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
sketch_graph_encoder = Encoders.gnn_full.gnn.SemanticModule()
sketch_graph_decoder = Encoders.gnn_full.gnn.Sketch_prediction()
sketch_graph_encoder.load_state_dict(torch.load(os.path.join(sketch_dir, 'graph_encoder.pth')))
sketch_graph_decoder.load_state_dict(torch.load(os.path.join(sketch_dir, 'graph_decoder.pth')))


def sketch_predict(gnn_graph):
    x_dict = sketch_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    output = sketch_graph_decoder(x_dict)

    selected_indices_raw = Models.sketch_arguments.face_aggregate.face_aggregate_withMask(gnn_graph.x_dict['stroke'], output)
    # selected_indices_raw = Models.sketch_arguments.face_aggregate.loop_chosing(gnn_graph.x_dict['stroke'], output)

    selected_indices = selected_indices_raw.bool().squeeze()
    selected_node_features = node_features[selected_indices]
    normal = Models.sketch_arguments.face_aggregate.sketch_to_normal(selected_node_features.tolist())
    
    Models.sketch_model_helper.vis_gt_strokes(node_features, selected_indices_raw)

    if selected_node_features.shape[0] == 0:
        print("failed to produce sketch")
        
    sketch_points = Models.sketch_arguments.face_aggregate.extract_unique_points(selected_node_features)
    return selected_indices_raw, sketch_points, normal


# Extrude Prediction
Extrude_dir = os.path.join(current_dir, 'checkpoints', 'extrude_prediction')
Extrude_encoder = Encoders.gnn_full.gnn.SemanticModule()
Extrude_decoder = Encoders.gnn_full.gnn.ExtrudingStrokePrediction()
Extrude_encoder.load_state_dict(torch.load(os.path.join(Extrude_dir, 'graph_encoder.pth')))
Extrude_decoder.load_state_dict(torch.load(os.path.join(Extrude_dir, 'graph_decoder.pth')))


def extrude_predict(gnn_graph, sketch_strokes, edge_features):
    x_dict = Extrude_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    strokes_indices = Extrude_decoder(x_dict, gnn_graph.edge_index_dict, sketch_strokes)
    extrude_amount, direction = Models.sketch_arguments.face_aggregate.get_extrude_amount(gnn_graph.x_dict['stroke'], strokes_indices, sketch_strokes, edge_features)
    return extrude_amount, direction


# --------------------- cascade_brep_features --------------------- #

def cascade_brep_features(brep_files):
    final_brep_edges = []
    prev_brep_edges = []

    for file_name in brep_files:
        brep_file_path = os.path.join(output_relative_dir, file_name)
        edge_features_list, edge_coplanar_list = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)
        
        if len(prev_brep_edges) == 0:
            final_brep_edges = edge_features_list
            prev_brep_edges = edge_features_list
        else:
            # We already have brep
            new_features, new_planes= Preprocessing.generate_dataset.find_new_features(prev_brep_edges, edge_features_list, edge_coplanar_list) 
            final_brep_edges += new_features
            prev_brep_edges = edge_features_list

    return final_brep_edges


# --------------------- Main Code --------------------- #

for batch in tqdm(data_loader):
    node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, edge_features, brep_coplanar, new_features= batch

    node_features = node_features.to(torch.float32).to(device).squeeze(0)
    edge_features = torch.empty(1, 0).to(torch.float32).to(device)


    # Program State Init
    current_brep_embedding = torch.empty((1, 0))
    current_brep_embedding = current_brep_embedding.to(torch.float32).to(device).squeeze(0)
    current__brep_class = Preprocessing.proc_CAD.generate_program.Brep()

    current_face_feature_gnn_list = torch.empty((1, 0))
    current_program = torch.tensor([], dtype=torch.int64)


    # Parser Init
    file_path = os.path.join(output_dir, 'Program.json')


    # Graph Init
    gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
    gnn_graph.set_brep_connection(edge_features)
    
    # Program Init
    next_op = Op_predict(gnn_graph, current_program)


    while next_op != 0:
        print("Op Executing", next_op)
        
        # Terminate
        if next_op == 0 or next_op == 3:
            break
                    
        # Sketch
        if next_op == 1:
            prev_sketch_index, sketch_points, normal= sketch_predict(gnn_graph)
            current__brep_class._sketch_op(sketch_points, normal, sketch_points.tolist())

        # Extrude
        if next_op == 2:
            extrude_amount, direction= extrude_predict(gnn_graph, prev_sketch_index, edge_features)
            current__brep_class.extrude_op(extrude_amount, direction.tolist())
        
        # Write the Program
        current__brep_class.write_to_json(output_dir)

        # Read the Program and produce brep file
        if os.path.exists(output_relative_dir):
            shutil.rmtree(output_relative_dir)
        os.makedirs(output_relative_dir, exist_ok=True)

        parsed_program_class = Preprocessing.proc_CAD.Program_to_STL.parsed_program(file_path, output_dir)
        parsed_program_class.read_json_file()

        # Read brep file
        brep_files = [file_name for file_name in os.listdir(os.path.join(output_dir, 'canvas'))
                if file_name.startswith('brep_') and file_name.endswith('.step')]
        brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        edge_features = cascade_brep_features(brep_files)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)



        # Update the graph
        gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
        gnn_graph.set_brep_connection(edge_features)

        # Predict next Operation
        if next_op == 1:
            next_op = 2
        else:
            next_op = 1
        # current_program = torch.cat((current_program, torch.tensor([next_op], dtype=torch.int64)))
        # next_op = Op_predict(gnn_graph, current_program)
        
        Models.sketch_model_helper.vis_stroke_cloud(edge_features)

        print("Next Op", next_op)
        print("------------")
    
    Models.sketch_model_helper.vis_stroke_cloud(node_features)
    break



