
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
graph_decoder = Encoders.gnn_full.gnn.Sketch_brep_prediction()

graph_encoder.to(device)
graph_decoder.to(device)

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'full_graph_sketch')
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
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/full_train_dataset')
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
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, face_boundary_points, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

            if edge_features.shape[1] == 0:
                continue
            
            # Move to device
            node_features = node_features.to(torch.float32).to(device).squeeze(0)
            operations_matrix = operations_matrix.to(torch.float32).to(device)
            intersection_matrix = intersection_matrix.to(torch.float32).to(device)
            operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)
            edge_features = edge_features.to(torch.float32).to(device).squeeze(0)
            
            # Create graph
            gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.set_brep_connection(edge_features, face_feature_gnn_list)
            
            # Forward pass
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict)
            
            # prepare gt
            target_op_index = len(program[0]) - 1
            op_to_index_matrix = operations_order_matrix
            kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
            chosen_mask = kth_operation.flatten() == 1
            chosen_node_matrix = node_features[chosen_mask]
            gt = Models.sketch_model_helper.chosen_edge_id(chosen_node_matrix, edge_features)

            if gt is None:
                continue
            
            loss = loss_function(output, gt)
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
                node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, face_boundary_points, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

                if edge_features.shape[1] == 0:
                    continue
                
                # Move to device
                node_features = node_features.to(torch.float32).to(device).squeeze(0)
                operations_matrix = operations_matrix.to(torch.float32).to(device)
                intersection_matrix = intersection_matrix.to(torch.float32).to(device)
                operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)
                edge_features = edge_features.to(torch.float32).to(device).squeeze(0)
                
                # Create graph
                gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
                gnn_graph.set_brep_connection(edge_features, face_feature_gnn_list)
                
                # Forward pass
                x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
                output = graph_decoder(x_dict)
                target_op_index = len(program[0]) - 1
                op_to_index_matrix = operations_order_matrix
                kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
                chosen_mask = kth_operation.flatten() == 1
                chosen_node_matrix = node_features[chosen_mask]
                gt = Models.sketch_model_helper.chosen_edge_id(chosen_node_matrix, edge_features)

                if gt is None:
                    continue
         
                # Compute loss
                loss = loss_function(output, gt)
                total_val_loss += loss.item()
        
        if best_val_loss > total_val_loss:
            best_val_loss =  total_val_loss
            save_models()
        
        # Print epoch losses
        print(f"Epoch {epoch+1}: Train Loss = {total_train_loss/len(train_loader)}, Val Loss = {total_val_loss/len(val_loader)}")



def eval():
    load_models()
    train_loader, val_loader = load_dataset()

    graph_encoder.eval()
    graph_decoder.eval()
    total_val_loss = 0.0
    
    total_edges = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, face_boundary_points, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

            # Skip batches with no edge features
            if edge_features.shape[1] == 0:
                continue

            node_features = node_features.to(torch.float32).to(device).squeeze(0)
            operations_matrix = operations_matrix.to(torch.float32).to(device)
            intersection_matrix = intersection_matrix.to(torch.float32).to(device)
            operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)
            edge_features = edge_features.to(torch.float32).to(device).squeeze(0)
            
            # Create the graph data structure for the GNN
            gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.set_brep_connection(edge_features, face_feature_gnn_list)
            
            # Perform a forward pass through the model to get the output
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict)

            # prepare gt
            target_op_index = len(program[0]) - 1
            op_to_index_matrix = operations_order_matrix
            kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
            chosen_mask = kth_operation.flatten() == 1
            chosen_node_matrix = node_features[chosen_mask]
            gt = Models.sketch_model_helper.chosen_edge_id(chosen_node_matrix, edge_features)

            if gt is None:
                continue
            
            print("out", output)
            print("gt", gt)
            # Vis
            gt_mask = (gt > 0).float()
            pred_mask = (output > 0.3).float()
            # Update total edges and correct predictions
            total_edges += 1

            if not torch.all(pred_mask == gt_mask):
                pass
                # Models.sketch_model_helper.vis_stroke_cloud(node_features)
                # Models.sketch_model_helper.vis_gt_strokes(edge_features, gt)
                # Models.sketch_model_helper.vis_gt_strokes(edge_features, output)
                # Models.sketch_model_helper.vis_stroke_cloud(node_features)
                # Models.sketch_model_helper.vis_gt_strokes(edge_features, gt)
                # Models.sketch_model_helper.vis_gt_strokes(edge_features, output)

            else:
                correct_predictions += 1

    # Compute the accuracy
    accuracy = correct_predictions / total_edges

    # Print the accuracy
    print(f"Accuracy: {accuracy * 100:.2f}%")


def predict_brep_edges(graph_encoder, graph_decoder, batch):

    node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, face_boundary_points, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

    node_features = node_features.to(torch.float32).to(device).squeeze(0)
    operations_matrix = operations_matrix.to(torch.float32).to(device)
    intersection_matrix = intersection_matrix.to(torch.float32).to(device)
    operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)
    edge_features = edge_features.to(torch.float32).to(device).squeeze(0)
    
    # Create the graph data structure for the GNN
    gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
    gnn_graph.set_brep_connection(edge_features, face_feature_gnn_list)
    
    # Perform a forward pass through the model to get the output
    x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    output = graph_decoder(x_dict)

    return output


#---------------------------------- Public Functions ----------------------------------#

# train()
