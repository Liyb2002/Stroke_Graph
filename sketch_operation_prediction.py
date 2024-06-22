import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network

import Encoders.gnn.gnn
import Encoders.program_encoder.program_encoder

import Models.sketch_model


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
SBGCN_model = Preprocessing.SBGCN.SBGCN_network.FaceEdgeVertexGCN()
stroke_embed_model = Models.sketch_model.StrokeEmbeddingNetwork()
face_embed_model = Models.sketch_model.PlaneEmbeddingNetwork()

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
os.makedirs(save_dir, exist_ok=True)


def load_models():
    pass

def save_models():
    pass

def train_face_prediction():

    # Define training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(SBGCN_model.parameters()),
        lr=0.001
    )

    epochs = 1

    # Create a DataLoader
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/example')

    # Filter to only keep
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
        SBGCN_model.train()
        stroke_embed_model.train()
        face_embed_model.train()
        
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, face_to_stroke, program, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch
            
            # 1) Embed the strokes
            if edge_features.shape[1] == 0: 
                edge_features = torch.zeros((1, 1, 6))

            edge_features = edge_features.to(torch.float32).to(device)
            stroke_embed = stroke_embed_model(edge_features)

            # 2) Pair each stroke with faces
            index_id = index_id[0]
            face_embed = face_embed_model(edge_index_face_edge_list, index_id, stroke_embed)
            print("face_embeddings", face_embed.shape)







#---------------------------------- Public Functions ----------------------------------#

train_face_prediction()
