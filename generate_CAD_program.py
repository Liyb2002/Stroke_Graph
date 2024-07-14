import Preprocessing.dataloader
import Preprocessing.gnn_graph_full


import Encoders.gnn_full.gnn

from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os

import random

# --------------------- Dataset --------------------- #
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/full_train_dataset')
good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 0]
filtered_dataset = Subset(dataset, good_data_indices)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


# --------------------- Directory --------------------- #
current_dir = os.getcwd()


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


# Sketch with brep Prediction
Sketch_with_brep_dir = os.path.join(current_dir, 'checkpoints', 'full_graph_sketch')
Sketch_with_brep_encoder = Encoders.gnn_full.gnn.SemanticModule()
Sketch_with_brep_decoder = Encoders.gnn_full.gnn.Sketch_brep_prediction()
Sketch_with_brep_encoder.load_state_dict(torch.load(os.path.join(Op_dir, 'graph_encoder.pth')))
Sketch_with_brep_decoder.load_state_dict(torch.load(os.path.join(Op_dir, 'graph_decoder.pth')))

Sketch_choosing_dir = os.path.join(current_dir, 'checkpoints', 'empty_brep_sketch')
Sketch_choosing_decoder = Encoders.gnn_full.gnn.Final_stroke_finding()
Sketch_choosing_decoder.load_state_dict(torch.load(os.path.join(Op_dir, 'strokes_decoder.pth')))

Sketch_empty_brep_dir = os.path.join(current_dir, 'checkpoints', 'empty_brep_sketch')
Sketch_empty_decoder = Encoders.gnn_full.gnn.Empty_brep_prediction()
Sketch_empty_decoder.load_state_dict(torch.load(os.path.join(Op_dir, 'graph_decoder.pth')))


def sketch_predict(gnn_graph, current_program):
    if len(current_program) == 0:
        x_dict = Sketch_empty_decoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
        output = Sketch_empty_decoder(x_dict)
    else:
        x_dict = Sketch_with_brep_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
        stroke_weights = Sketch_with_brep_decoder(x_dict)
        output = Sketch_choosing_decoder(x_dict, gnn_graph.edge_index_dict, stroke_weights)
    
    return output



# --------------------- Main Code --------------------- #

for batch in tqdm(data_loader):
    node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, face_boundary_points, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

    # Stroke Cloud data process
    node_features = node_features.to(torch.float32).to(device).squeeze(0)
    operations_matrix = operations_matrix.to(torch.float32).to(device)
    intersection_matrix = intersection_matrix.to(torch.float32).to(device)
    operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)


    # Program State init
    current_brep = torch.empty((1, 0))
    current_brep = current_brep.to(torch.float32).to(device).squeeze(0)
    current_face_feature_gnn_list = torch.empty((1, 0))
    current_program = torch.tensor([], dtype=torch.int64)

    # Graph init
    gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
    gnn_graph.set_brep_connection(edge_features, current_face_feature_gnn_list)
    next_op = Op_predict(gnn_graph, current_program)

    while next_op != 0:
        print("next_op", next_op)
        
        if next_op == 1:
            sketches_prob = sketch_predict(gnn_graph, current_program)
            print("sketches_prob", sketches_prob.shape)

        # Predict next Operation
        current_program.append(next_op)
        next_op = Op_predict(gnn_graph, current_program)
    break