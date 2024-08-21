import Preprocessing.dataloader
import Preprocessing.gnn_graph_full
import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network

import Models.sketch_model_helper
import Encoders.gnn_full.gnn
import Models.sketch_arguments.face_aggregate

from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


graph_encoder = Encoders.gnn_full.gnn.SemanticModule()
graph_decoder = Encoders.gnn_full.gnn.ExtrudingStrokePrediction()

# Define optimizer and loss function
optimizer = optim.Adam( list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)
loss_function = nn.BCELoss()

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'extrude_prediction')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    # Load models if they exist
    if os.path.exists(os.path.join(save_dir, 'graph_encoder.pth')):
        graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth')))
        print("Loaded graph_encoder")

    if os.path.exists(os.path.join(save_dir, 'graph_decoder.pth')):
        graph_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth')))
        print("Loaded graph_decoder")

def save_models():
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))
    torch.save(graph_decoder.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))
    print("Saved models.")

def load_dataset():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/extrude_only_test')
    good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 2]
    filtered_dataset = Subset(dataset, good_data_indices)
    print(f"Total number of sketch data: {len(filtered_dataset)}")

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(filtered_dataset))
    val_size = len(filtered_dataset) - train_size
    train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_loader, val_loader


def load_eval_dataset():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/extrude_only_eval')
    good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 2]
    filtered_dataset = Subset(dataset, good_data_indices)
    print(f"Total number of sketch data: {len(filtered_dataset)}")

    eval_loader = DataLoader(filtered_dataset, batch_size=1, shuffle=False)

    return eval_loader


def train():
    train_loader, val_loader = load_dataset()
    # Training and validation loop
    best_val_loss = float('inf')
    epochs = 30

    for epoch in range(epochs):
        graph_encoder.train()
        graph_decoder.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, edge_features, brep_coplanar, new_features= batch
        
            node_features = node_features.to(torch.float32).to(device).squeeze(0)
            edge_features = torch.tensor(edge_features, dtype=torch.float32)

            # Prev sketch
            sketch_op_index = len(program[0]) - 2
            sketch_strokes = Models.sketch_arguments.face_aggregate.get_kth_operation(operations_order_matrix, sketch_op_index).to(device).to(torch.float32)

            # Current extrude
            target_op_index = len(program[0]) - 1
            kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(operations_order_matrix, target_op_index).to(device)
            extrude_strokes = Models.sketch_model_helper.choose_extrude_strokes(sketch_strokes, kth_operation, node_features)
            # extrude_opposite_face = Models.sketch_model_helper.choose_extrude_opposite_face(sketch_strokes, kth_operation, node_features)

            # Create graph
            gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.set_brep_connection(edge_features)

            # Forward pass
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict, gnn_graph.edge_index_dict, sketch_strokes)
            
            # Models.sketch_model_helper.vis_gt_strokes(node_features, sketch_strokes)
            # Models.sketch_model_helper.vis_gt_strokes(node_features, extrude_strokes)
            # Models.sketch_model_helper.vis_gt_strokes(node_features, extrude_opposite_face)


            loss = loss_function(output, extrude_strokes)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)



        # Validation loop
        graph_encoder.eval()
        graph_decoder.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, edge_features, brep_coplanar, new_features= batch
            
                node_features = node_features.to(torch.float32).to(device).squeeze(0)
                edge_features = torch.tensor(edge_features, dtype=torch.float32)

                # Prev sketch
                sketch_op_index = len(program[0]) - 2
                sketch_strokes = Models.sketch_arguments.face_aggregate.get_kth_operation(operations_order_matrix, sketch_op_index).to(device).to(torch.float32)

                # Current extrude
                target_op_index = len(program[0]) - 1
                kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(operations_order_matrix, target_op_index).to(device)
                extrude_strokes = Models.sketch_model_helper.choose_extrude_strokes(sketch_strokes, kth_operation, node_features)
                # extrude_opposite_face = Models.sketch_model_helper.choose_extrude_opposite_face(sketch_strokes, kth_operation, node_features)

                # Create graph
                gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
                gnn_graph.set_brep_connection(edge_features)

                # Forward pass
                x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
                output = graph_decoder(x_dict, gnn_graph.edge_index_dict, sketch_strokes)

                loss = loss_function(output, extrude_strokes)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        if best_val_loss > total_val_loss:
            best_val_loss =  total_val_loss
            save_models()

        print(f"Epoch {epoch+1}/{epochs}: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")


#---------------------------------- Public Functions ----------------------------------#

train()
