
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
graph_model = Encoders.gnn_full.gnn.InstanceModule()
graph_model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(graph_model.parameters(), lr=0.001)
loss_function = nn.BCELoss()

# Load the dataset
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/train_dataset')
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

# Training and validation loop
best_val_loss = float('inf')

for epoch in range(10):  # Assuming you want to train for 10 epochs
    # Training loop
    graph_model.train()
    total_train_loss = 0.0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10 - Training"):
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
        output = graph_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)
        gt = Models.sketch_model_helper.chosen_edge_id(face_boundary_points[len(program[0])-1], edge_features)

        # Compute loss and backpropagate
        loss = loss_function(output, gt)  # Define your loss function
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_train_loss += loss.item()
    
    # Validation loop
    graph_model.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/10 - Validation"):
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
            output = graph_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            gt = Models.sketch_model_helper.chosen_edge_id(face_boundary_points[len(program[0])-1], edge_features)

            # Compute loss
            loss = loss_function(output, gt)
            total_val_loss += loss.item()
    
    # Print epoch losses
    print(f"Epoch {epoch+1}: Train Loss = {total_train_loss/len(train_loader)}, Val Loss = {total_val_loss/len(val_loader)}")
