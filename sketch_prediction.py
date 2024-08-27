
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


# Initialize your model and move it to the device
graph_encoder = Encoders.gnn_full.gnn.SemanticModule()
graph_decoder = Encoders.gnn_full.gnn.Sketch_prediction()

graph_encoder.to(device)
graph_decoder.to(device)

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
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



# Define optimizer and loss function
optimizer = optim.Adam( list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)
loss_function = nn.BCELoss()

def load_dataset():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/extrude_only_test')
    good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 1]
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
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/CAD2Sketch')
    good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 1]
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
        # Training loop
        graph_encoder.train()
        graph_decoder.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, edge_features, brep_coplanar, new_features= batch
        
            node_features = node_features.to(torch.float32).to(device).squeeze(0)
            edge_features = torch.tensor(edge_features, dtype=torch.float32)

            # Now build graph
            gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.set_brep_connection(edge_features)

            # Forward pass
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict)
            
            # prepare gt
            target_op_index = len(program[0]) - 1
            op_to_index_matrix = operations_order_matrix
            kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, target_op_index).to(device).to(torch.float32)

            # Models.sketch_model_helper.vis_gt_strokes(node_features, kth_operation)
                
            loss = loss_function(output, kth_operation)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
        
        # Validation loop
        graph_encoder.eval()
        graph_decoder.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, edge_features, brep_coplanar, new_features= batch
            
                node_features = node_features.to(torch.float32).to(device).squeeze(0)
                edge_features = torch.tensor(edge_features, dtype=torch.float32)

                # Now build graph
                gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
                gnn_graph.set_brep_connection(edge_features)

                # Forward pass
                x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
                output = graph_decoder(x_dict)
                
                # prepare gt
                target_op_index = len(program[0]) - 1
                op_to_index_matrix = operations_order_matrix
                kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, target_op_index).to(device).to(torch.float32)
            
                # Compute loss
                loss = loss_function(output, kth_operation)
                total_val_loss += loss.item()
        
        if best_val_loss > total_val_loss:
            best_val_loss =  total_val_loss
            save_models()
        
        # Print epoch losses
        print(f"Epoch {epoch+1}: Train Loss = {total_train_loss/len(train_loader)}, Val Loss = {total_val_loss/len(val_loader)}")



def eval():
    load_models()
    eval_loader = load_eval_dataset()

    graph_encoder.eval()
    graph_decoder.eval()
    total_val_loss = 0.0
    
    total_predictions = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, edge_features, brep_coplanar, new_features= batch
        
            node_features = node_features.to(torch.float32).to(device).squeeze(0)
            edge_features = torch.tensor(edge_features, dtype=torch.float32)

            # Now build graph
            gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.set_brep_connection(edge_features)

            # Forward pass
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict)
            
            # prepare gt
            target_op_index = len(program[0]) - 1
            op_to_index_matrix = operations_order_matrix
            kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, target_op_index).to(device).to(torch.float32)
            Models.sketch_model_helper.vis_stroke_cloud(node_features)

            if len(program[0]) > 3:
                output_binary = (output > 0.5).float()

                # Ensure gt and output are the same shape
                kth_operation = kth_operation.view(-1, 1)
                output_binary = output_binary.view(-1, 1)
                if torch.equal(output_binary, kth_operation):
                    correct_predictions += 1
                # else:
                #     Models.sketch_model_helper.vis_stroke_cloud(node_features)
                #     Models.sketch_model_helper.vis_stroke_cloud(edge_features)
                #     Models.sketch_model_helper.vis_gt_strokes(node_features, kth_operation)
                    # Models.sketch_model_helper.vis_gt_strokes(node_features, output)

                total_predictions += 1


    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Number Predictions: {total_predictions:.4f}")
    print(f"Evaluation Accuracy: {accuracy:.4f}")



def predict_brep_edges(graph_encoder, graph_decoder, batch):

    node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, edge_features, brep_coplanar, new_features= batch

    node_features = node_features.to(torch.float32).to(device).squeeze(0)
    edge_features = torch.tensor(edge_features, dtype=torch.float32)

    # Now build graph
    gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
    gnn_graph.set_brep_connection(edge_features, brep_coplanar)

    # Forward pass
    x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    output = graph_decoder(x_dict, gnn_graph.edge_index_dict)

    return output


#---------------------------------- Public Functions ----------------------------------#

eval()
