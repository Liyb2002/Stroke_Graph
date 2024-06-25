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
 
# SBGCN_model = Preprocessing.SBGCN.SBGCN_network.FaceEdgeVertexGCN()
StrokeEmbeddingNetwork = Models.sketch_model.StrokeEmbeddingNetwork()
graph_embedding_model = Encoders.gnn.gnn.SemanticModule()
BrepStrokeCloudAttention = Models.sketch_model.BrepStrokeCloudAttention()
BrepFaceEdgeAttention = Models.sketch_model.BrepStrokeCloudAttention()

StrokeEmbeddingNetwork.to(device)
graph_embedding_model.to(device)
BrepStrokeCloudAttention.to(device)
BrepFaceEdgeAttention.to(device)

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
    torch.save(SBGCN_model.state_dict(), os.path.join(save_dir, 'SBGCN_model.pth'))
    torch.save(graph_embedding_model.state_dict(), os.path.join(save_dir, 'graph_embedding_model.pth'))
    torch.save(BrepStrokeCloudAttention.state_dict(), os.path.join(save_dir, 'BrepStrokeCloudAttention.pth'))
    torch.save(BrepFaceEdgeAttention.state_dict(), os.path.join(save_dir, 'BrepFaceEdgeAttention.pth'))

    print("Saved face models.")



def train_face_prediction():

    # Define training
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        list(StrokeEmbeddingNetwork.parameters())+
        list(graph_embedding_model.parameters())+
        list(BrepStrokeCloudAttention.parameters()),
        lr= 1e-4
    )

    epochs = 20

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
        StrokeEmbeddingNetwork.train()
        graph_embedding_model.train()
        BrepStrokeCloudAttention.train()
        # BrepFaceEdgeAttention.train()
        
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, face_to_stroke, program, face_boundary_points, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch
            
            # 1) Prepare the brep embedding
            if edge_features.shape[1] == 0: 
                continue
                edge_features = torch.zeros((1, 1, 6))

            # brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
            #             edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
            # brep_graph.to_device(device)
            # face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)

            edge_embedding = StrokeEmbeddingNetwork(edge_features)

            
            # 2) Prepare the stroke cloud embedding
            node_features = node_features.to(torch.float32).to(device)
            operations_matrix = operations_matrix.to(torch.float32).to(device)
            intersection_matrix = intersection_matrix.to(torch.float32).to(device)
            operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)

            # graph embedding
            gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.to_device(device)
            graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)


            # 3) Cross attention on edge_embedding and stroke cloud
            # attentioned_edge is the edge embedding of the brep while contains information from the stroke cloud
            # attentioned_edge has shape (1, num_edges, 32)
            # attentioned_vertex has shape (1, num_vertex, 32)
            edge_probs = BrepStrokeCloudAttention(edge_embedding, graph_embedding)



            # 5) Prepare the gt_matrix
            operation_count = len(program[0]) -1 
            boundary_points = face_boundary_points[operation_count]
            gt_matrix = Models.sketch_model_helper.chosen_edge_id(boundary_points, edge_features)


            # 6) Calculate loss and update weights
            loss = criterion(edge_probs, gt_matrix)

            total_train_loss += loss.item()

            if epoch > 10:
                Models.sketch_model_helper.vis_stroke_cloud(node_features)
                Models.sketch_model_helper.vis_gt_strokes(edge_features, gt_matrix)
                Models.sketch_model_helper.vis_predicted_strokes(edge_features, edge_probs)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}")




#---------------------------------- Public Functions ----------------------------------#


train_face_prediction()
