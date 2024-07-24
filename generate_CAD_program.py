import Preprocessing.dataloader
import Preprocessing.gnn_graph_Gen

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
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/full_train_dataset')
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
Sketch_with_brep_dir = os.path.join(current_dir, 'checkpoints', 'full_graph_sketch')
Sketch_with_brep_encoder = Encoders.gnn_full.gnn.SemanticModule()
Sketch_with_brep_decoder = Encoders.gnn_full.gnn.Sketch_brep_prediction()
Sketch_with_brep_encoder.load_state_dict(torch.load(os.path.join(Sketch_with_brep_dir, 'graph_encoder.pth')))
Sketch_with_brep_decoder.load_state_dict(torch.load(os.path.join(Sketch_with_brep_dir, 'graph_decoder.pth')))

Sketch_choosing_dir = os.path.join(current_dir, 'checkpoints', 'stroke_choosing')
Sketch_choosing_decoder = Encoders.gnn_full.gnn.Final_stroke_finding()
Sketch_choosing_decoder.load_state_dict(torch.load(os.path.join(Sketch_choosing_dir, 'strokes_decoder.pth')))

Sketch_empty_brep_dir = os.path.join(current_dir, 'checkpoints', 'empty_brep_sketch')
Sketch_empty_encoder = Encoders.gnn_full.gnn.SemanticModule()
Sketch_empty_decoder = Encoders.gnn_full.gnn.Empty_brep_prediction()
Sketch_empty_encoder.load_state_dict(torch.load(os.path.join(Sketch_empty_brep_dir, 'graph_encoder.pth')))
Sketch_empty_decoder.load_state_dict(torch.load(os.path.join(Sketch_empty_brep_dir, 'graph_decoder.pth')))


def sketch_predict(gnn_graph, current_program, node_features, brep_stroke_connection_matrix, stroke_coplanar_matrix):
    if len(current_program) == 0:
        x_dict = Sketch_empty_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
        output = Sketch_empty_decoder(x_dict)
    else:
        x_dict = Sketch_with_brep_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
        brep_edges_weights = Sketch_with_brep_decoder(x_dict, gnn_graph.edge_index_dict)
        stroke_weights = Models.sketch_model_helper.integrate_brep_probs(brep_edges_weights, brep_stroke_connection_matrix, stroke_coplanar_matrix)
        
        x_dict = Sketch_with_brep_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
        output = Sketch_choosing_decoder(x_dict, gnn_graph.edge_index_dict, stroke_weights)
    
        Models.sketch_model_helper.vis_stroke_cloud(node_features)
        Models.sketch_model_helper.vis_stroke_cloud(gnn_graph.x_dict['brep'])
        Models.sketch_model_helper.vis_gt_strokes(gnn_graph.x_dict['brep'], brep_edges_weights)
        Models.sketch_model_helper.vis_gt_strokes(node_features, output)

    selected_indices_raw = Models.sketch_arguments.face_aggregate.face_aggregate_withMask(node_features, output)
    selected_indices = selected_indices_raw.bool().squeeze()
    selected_node_features = node_features[selected_indices]
    normal = Models.sketch_arguments.face_aggregate.sketch_to_normal(selected_node_features.tolist())
    
    if selected_node_features.shape[0] == 0:
        print("failed to produce sketch")
        
    sketch_points = Models.sketch_arguments.face_aggregate.extract_unique_points(selected_node_features)
    return selected_indices_raw, sketch_points, normal


# Extrude Prediction
Extrude_dir = os.path.join(current_dir, 'checkpoints', 'extrude')
Extrude_encoder = Encoders.gnn_full.gnn.SemanticModule()
Extrude_decoder = Encoders.gnn_full.gnn.ExtrudingStrokePrediction()
Extrude_encoder.load_state_dict(torch.load(os.path.join(Extrude_dir, 'graph_encoder.pth')))
Extrude_decoder.load_state_dict(torch.load(os.path.join(Extrude_dir, 'graph_decoder.pth')))


def extrude_predict(gnn_graph, sketch_strokes, node_features):
    x_dict = Extrude_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    strokes_indices = Extrude_decoder(x_dict, gnn_graph.edge_index_dict, sketch_strokes)
    extrude_amount, direction = Models.sketch_arguments.face_aggregate.get_extrude_amount(node_features, strokes_indices, sketch_strokes)
    return extrude_amount, direction

# --------------------- Main Code --------------------- #

for batch in tqdm(data_loader):
    node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, face_boundary_points, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

    # Stroke Cloud data process
    node_features = node_features.to(torch.float32).to(device).squeeze(0)
    operations_matrix = operations_matrix.to(torch.float32).to(device)
    intersection_matrix = intersection_matrix.to(torch.float32).to(device)
    operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)


    # Program State init
    current_brep_embedding = torch.empty((1, 0))
    current_brep_embedding = current_brep_embedding.to(torch.float32).to(device).squeeze(0)
    current__brep_class = Preprocessing.proc_CAD.generate_program.Brep()

    current_face_feature_gnn_list = torch.empty((1, 0))
    current_program = torch.tensor([], dtype=torch.int64)


    # Parser Init
    file_path = os.path.join(output_dir, 'Program.json')


    # Graph init
    gnn_graph = Preprocessing.gnn_graph_Gen.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
    gnn_graph.set_brep_connection(current_brep_embedding, current_face_feature_gnn_list)
    next_op = Op_predict(gnn_graph, current_program)

    brep_stroke_connection_matrix = None
    stroke_coplanar_matrix = None


    while next_op != 0:
        print("Op Executing", next_op)
        
        # Terminate
        if next_op == 0 or next_op == 3:
            break
        
        # Sketch
        if next_op == 1:
            prev_sketch_index, sketch_points, normal= sketch_predict(gnn_graph, current_program, node_features, brep_stroke_connection_matrix, stroke_coplanar_matrix)
            current__brep_class._sketch_op(sketch_points, normal, sketch_points.tolist())

        # Extrude
        if next_op == 2:
            extrude_amount, direction= extrude_predict(gnn_graph, prev_sketch_index, node_features)
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
        brep_file_path = brep_files[-1]
        brep_file_path = os.path.join(output_relative_dir, brep_file_path)
        face_feature_gnn_list, face_features_list, edge_features_list, vertex_features_list, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id= Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)

        # Update the graph
        gnn_graph = Preprocessing.gnn_graph_Gen.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
        brep_stroke_connection_matrix, stroke_coplanar_matrix = gnn_graph.set_brep_connection(edge_features_list, face_feature_gnn_list)


        # Predict next Operation
        current_program = torch.cat((current_program, torch.tensor([next_op], dtype=torch.int64)))
        next_op = Op_predict(gnn_graph, current_program)
        print("Next Op", next_op)
        print("------------")
    
    Models.sketch_model_helper.vis_stroke_cloud(node_features)
    break
