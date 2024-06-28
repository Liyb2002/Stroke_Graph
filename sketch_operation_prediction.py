import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network

import Encoders.gnn.gnn
import Encoders.program_encoder.program_encoder

import Models.sketch_model
import Models.sketch_model_helper

import Models.sketch_arguments.face_aggregate

from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from itertools import combinations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the neural networks
 
# Stroke_embedding_model = Models.sketch_model.StrokeEmbeddingNetwork()
SBGCN_model = Preprocessing.SBGCN.SBGCN_network.FaceEdgeVertexGCN()
# SelfAttentionModel = Models.sketch_model.SelfAttentionModel()
graph_embedding_model = Encoders.gnn.gnn.SemanticModule()
BrepStrokeCloudAttention = Models.sketch_model.BrepStrokeCloudAttention()
# BrepFaceEdgeAttention = Models.sketch_model.BrepStrokeCloudAttention()

graph_embedding_model.to(device)
SBGCN_model.to(device)
# SelfAttentionModel.to(device)
# graph_embedding_model.to(device)
BrepStrokeCloudAttention.to(device)
# BrepFaceEdgeAttention.to(device)

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
os.makedirs(save_dir, exist_ok=True)



def load_face_models():
    # Load models if they exist
    if os.path.exists(os.path.join(save_dir, 'SBGCN_model.pth')):
        SBGCN_model.load_state_dict(torch.load(os.path.join(save_dir, 'SBGCN_model.pth')))
        print("Loaded SBGCN_model")

    if os.path.exists(os.path.join(save_dir, 'graph_embedding_model.pth')):
        graph_embedding_model.load_state_dict(torch.load(os.path.join(save_dir, 'graph_embedding_model.pth')))
        print("Loaded graph_embedding_model")

    if os.path.exists(os.path.join(save_dir, 'BrepStrokeCloudAttention.pth')):
        BrepStrokeCloudAttention.load_state_dict(torch.load(os.path.join(save_dir, 'BrepStrokeCloudAttention.pth')))
        print("Loaded BrepStrokeCloudAttention")
    
    if os.path.exists(os.path.join(save_dir, 'BrepFaceEdgeAttention.pth')):
        BrepFaceEdgeAttention.load_state_dict(torch.load(os.path.join(save_dir, 'BrepFaceEdgeAttention.pth')))
        print("Loaded BrepFaceEdgeAttention")


def save_face_models():
    torch.save(graph_embedding_model.state_dict(), os.path.join(save_dir, 'graph_embedding_model.pth'))
    torch.save(SBGCN_model.state_dict(), os.path.join(save_dir, 'SBGCN_model.pth'))
    torch.save(BrepStrokeCloudAttention.state_dict(), os.path.join(save_dir, 'SelfAttentionModel.pth'))
    # torch.save(BrepFaceEdgeAttention.state_dict(), os.path.join(save_dir, 'BrepFaceEdgeAttention.pth'))

    print("Saved face models.")



def train_face_prediction():

    # Define training
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        list(graph_embedding_model.parameters())+
        list(SBGCN_model.parameters())+
        list(BrepStrokeCloudAttention.parameters()),
        lr= 1e-3
    )

    epochs = 100

    # Create a DataLoader
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/example')

    # Filter to only keep good data
    good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 1]
    filtered_dataset = Subset(dataset, good_data_indices)
    print(f"Total number of sketch data: {len(filtered_dataset)}")

    # Split dataset into training and validation
    train_size = int(0.8 * len(filtered_dataset))
    val_size = len(filtered_dataset) - train_size
    train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        graph_embedding_model.train()
        SBGCN_model.train()
        BrepStrokeCloudAttention.train()
        
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, face_to_stroke, program, face_boundary_points, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch
            
            # 1) Prepare the brep embedding
            if edge_features.shape[1] == 0: 
                continue
                edge_features = torch.zeros((1, 1, 6))

            brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
                        edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
            brep_graph.to_device(device)
            face_embedding, brep_edge_embedding, vertex_embedding = SBGCN_model(brep_graph)

            
            # 2) Prepare the stroke cloud embedding
            node_features = node_features.to(torch.float32).to(device)
            operations_matrix = operations_matrix.to(torch.float32).to(device)
            intersection_matrix = intersection_matrix.to(torch.float32).to(device)
            operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)

            # graph embedding
            gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.to_device(device)
            stroke_cloud_graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)


            # 3) Cross attention on edge_embedding and stroke cloud
            edge_left = BrepStrokeCloudAttention(brep_edge_embedding, stroke_cloud_graph_embedding)


            # 4) Prepare the gt_matrix
            gt_left = Models.sketch_model_helper.find_left_edge(edge_features, node_features)


            # 5) Calculate validation loss
            loss = criterion(edge_left, gt_left)

            total_train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}")
        




#---------------------------------- Public Functions ----------------------------------#


train_face_prediction()