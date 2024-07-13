import Preprocessing.dataloader
import Preprocessing.gnn_graph_full
import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network

import Models.sketch_model_helper
import Encoders.gnn_full.gnn
import Models.sketch_arguments.face_aggregate

import full_graph_train

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



# Define optimizer and loss function
# optimizer = optim.Adam( strokes_decoder.parameters(), lr=0.0004)
loss_function = nn.BCELoss()

# Load the dataset
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/full_train_dataset')
good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 2]
filtered_dataset = Subset(dataset, good_data_indices)
print(f"Total number of extrude data: {len(filtered_dataset)}")

# Split the dataset into training and validation sets
train_size = int(0.8 * len(filtered_dataset))
val_size = len(filtered_dataset) - train_size
train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)



def train():
    # Training and validation loop
    best_val_loss = float('inf')
    epochs = 30

    for epoch in range(epochs):
        # Training loop
        # strokes_decoder.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, operations_matrix, intersection_matrix, op_to_index_matrix, _, program, face_boundary_points, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

            node_features = node_features.to(torch.float32).to(device).squeeze(0)
            operations_matrix = operations_matrix.to(torch.float32).to(device)
            intersection_matrix = intersection_matrix.to(torch.float32).to(device)
            op_to_index_matrix = op_to_index_matrix.to(torch.float32).to(device)
            edge_features = edge_features.to(torch.float32).to(device).squeeze(0)

            # Prev sketch
            sketch_op_index = len(program[0]) - 2
            sketch_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, sketch_op_index).to(device)

            # Current extrude
            target_op_index = len(program[0]) - 1
            kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
            extrude_strokes = Models.sketch_model_helper.choose_extrude_strokes(sketch_operation, kth_operation, node_features)

            print("sketch_operation", sketch_operation.shape)
            print("node_features", node_features.shape)
            Models.sketch_model_helper.vis_gt_strokes(node_features, sketch_operation)
            Models.sketch_model_helper.vis_gt_strokes(node_features, extrude_strokes)



#---------------------------------- Public Functions ----------------------------------#

train()
