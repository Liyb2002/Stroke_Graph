import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network

import Encoders.gnn.gnn
import Encoders.program_encoder.program_encoder

import Models.sketch_model
import Models.sketch_model_helper

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
stroke_embed_model = Models.sketch_model.StrokeEmbeddingNetwork()
face_embed_model = Models.sketch_model.PlaneEmbeddingNetwork()
graph_embedding_model = Encoders.gnn.gnn.SemanticModule()
cross_attention_model = Models.sketch_model.FaceBrepAttention()

stroke_embed_model.to(device)
face_embed_model.to(device)
graph_embedding_model.to(device)
cross_attention_model.to(device)

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
os.makedirs(save_dir, exist_ok=True)



def load_models():
    # Load models if they exist
    if os.path.exists(os.path.join(save_dir, 'stroke_embed_model.pth')):
        stroke_embed_model.load_state_dict(torch.load(os.path.join(save_dir, 'stroke_embed_model.pth')))
        print("Loaded stroke_embed_model")

    if os.path.exists(os.path.join(save_dir, 'face_embed_model.pth')):
        face_embed_model.load_state_dict(torch.load(os.path.join(save_dir, 'face_embed_model.pth')))
        print("Loaded face_embed_model")

    if os.path.exists(os.path.join(save_dir, 'graph_embedding_model.pth')):
        graph_embedding_model.load_state_dict(torch.load(os.path.join(save_dir, 'graph_embedding_model.pth')))
        print("Loaded graph_embedding_model")

    if os.path.exists(os.path.join(save_dir, 'cross_attention_model.pth')):    
        cross_attention_model.load_state_dict(torch.load(os.path.join(save_dir, 'cross_attention_model.pth')))
        print("Loaded cross_attention_model")


def save_models():
    torch.save(stroke_embed_model.state_dict(), os.path.join(save_dir, 'stroke_embed_model.pth'))
    torch.save(face_embed_model.state_dict(), os.path.join(save_dir, 'face_embed_model.pth'))
    torch.save(graph_embedding_model.state_dict(), os.path.join(save_dir, 'graph_embedding_model.pth'))
    torch.save(cross_attention_model.state_dict(), os.path.join(save_dir, 'cross_attention_model.pth'))

    print("Saved models.")


def train_face_prediction():

    # Define training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(graph_embedding_model.parameters()) +
        list(stroke_embed_model.parameters()) +
        list(face_embed_model.parameters()) +
        list(cross_attention_model.parameters()),
        lr=0.001
    )

    epochs = 10

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
        stroke_embed_model.train()
        face_embed_model.train()
        graph_embedding_model.train()
        cross_attention_model.train()
        
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, face_to_stroke, program, face_boundary_points, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch
            
            # 1) Embed the strokes
            if edge_features.shape[1] == 0: 
                edge_features = torch.zeros((1, 1, 6))

            edge_features = edge_features.to(torch.float32).to(device)
            stroke_embed = stroke_embed_model(edge_features)


            # 2) Pair each stroke with faces
            index_id = index_id[0]
            face_embed = face_embed_model(edge_index_face_edge_list, index_id, stroke_embed)


            # 3) Prepare the stroke cloud embedding
            node_features = node_features.to(torch.float32).to(device)
            operations_matrix = operations_matrix.to(torch.float32).to(device)
            intersection_matrix = intersection_matrix.to(torch.float32).to(device)
            operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)

            gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.to_device(device)
            graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)


            # 4) Cross attention on face_embedding and stroke_cloud_embedding
            output = cross_attention_model(face_embed, graph_embedding)


            # 5) Build gt_matrix
            operation_count = len(program[0]) -1 
            boundary_points = face_boundary_points[operation_count]
            gt_matrix = Models.sketch_model_helper.chosen_face_id(boundary_points, edge_index_face_edge_list, index_id, edge_features)
            
            # 6) Calculate loss and update weights
            output = output.view(-1)
            gt_matrix = gt_matrix.view(-1).to(torch.float32).to(device)

            loss = criterion(output, gt_matrix)
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}")

        # Validation
        stroke_embed_model.eval()
        face_embed_model.eval()
        graph_embedding_model.eval()
        cross_attention_model.eval()

        total_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                node_features, operations_matrix, intersection_matrix, operations_order_matrix, face_to_stroke, program, face_boundary_points, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch
                
                if edge_features.shape[1] == 0: 
                    edge_features = torch.zeros((1, 1, 6))

                edge_features = edge_features.to(torch.float32).to(device)
                stroke_embed = stroke_embed_model(edge_features)

                index_id = index_id[0]
                face_embed = face_embed_model(edge_index_face_edge_list, index_id, stroke_embed)

                node_features = node_features.to(torch.float32).to(device)
                operations_matrix = operations_matrix.to(torch.float32).to(device)
                intersection_matrix = intersection_matrix.to(torch.float32).to(device)
                operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)

                gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
                gnn_graph.to_device(device)
                graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)

                output = cross_attention_model(face_embed, graph_embedding)

                operation_count = len(program[0]) -1 
                boundary_points = face_boundary_points[operation_count]
                gt_matrix = Models.sketch_model_helper.chosen_face_id(boundary_points, edge_index_face_edge_list, index_id, edge_features)

                output = output.view(-1)
                gt_matrix = gt_matrix.view(-1).to(torch.float32).to(device)

                loss = criterion(output, gt_matrix)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_models()




#---------------------------------- Public Functions ----------------------------------#




train_face_prediction()
