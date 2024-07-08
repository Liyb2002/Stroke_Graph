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


brep_graph_encoder = Encoders.gnn_full.gnn.SemanticModule()
brep_graph_decoder = Encoders.gnn_full.gnn.sketch_brep_prediction()


current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'stroke_choosing')
os.makedirs(save_dir, exist_ok=True)

def load_brep_models():
    # Load models if they exist
    brep_model_dir = os.path.join(current_dir, 'checkpoints', 'full_graph_sketch')

    if os.path.exists(os.path.join(brep_model_dir, 'graph_encoder.pth')):
        brep_graph_encoder.load_state_dict(torch.load(os.path.join(brep_model_dir, 'graph_encoder.pth')))
        print("Loaded graph_encoder")

    if os.path.exists(os.path.join(brep_model_dir, 'graph_decoder.pth')):
        brep_graph_decoder.load_state_dict(torch.load(os.path.join(brep_model_dir, 'graph_decoder.pth')))
        print("Loaded graph_decoder")



# Define optimizer and loss function
# optimizer = optim.Adam( list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)
loss_function = nn.BCELoss()

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




def train():
    load_brep_models()
    # Training and validation loop
    best_val_loss = float('inf')
    epochs = 30

    for epoch in range(epochs):
        # Training loop
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, face_boundary_points, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

            if edge_features.shape[1] == 0:
                continue
            
            # get the predictions 
            brep_edges_weights = full_graph_train.predict_brep_edges(brep_graph_encoder, brep_graph_decoder, batch)

            # move tp device
            node_features = node_features.to(torch.float32).to(device).squeeze(0)
            operations_matrix = operations_matrix.to(torch.float32).to(device)
            intersection_matrix = intersection_matrix.to(torch.float32).to(device)
            operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)
            edge_features = edge_features.to(torch.float32).to(device).squeeze(0)


            gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            brep_stroke_connection_matrix, stroke_coplanar_matrix = gnn_graph.set_brep_connection(edge_features, face_feature_gnn_list)

            stroke_weights = Models.sketch_model_helper.integrate_brep_probs(brep_edges_weights, brep_stroke_connection_matrix, stroke_coplanar_matrix)

            print("------")
#---------------------------------- Public Functions ----------------------------------#

train()


