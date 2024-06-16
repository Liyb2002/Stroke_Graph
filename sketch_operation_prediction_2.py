import Models.operation_model
import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network

import Models.sketch_arguments.face_aggregate
import Models.sketch_arguments.sketch_model_2


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


SBGCN_model = Preprocessing.SBGCN.SBGCN_network.FaceEdgeVertexGCN()
stroke_embed_model = Models.sketch_arguments.sketch_model_2.StrokeEmbeddingNetwork()
plane_embed_model = Models.sketch_arguments.sketch_model_2.PlaneEmbeddingNetwork()
cross_attention_model = Models.sketch_arguments.sketch_model_2.FaceBrepAttention()

SBGCN_model.to(device)
stroke_embed_model.to(device)
plane_embed_model.to(device)
cross_attention_model.to(device)

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction_2')
os.makedirs(save_dir, exist_ok=True)


def train():

    # Define training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(SBGCN_model.parameters())+
        list(stroke_embed_model.parameters())+
        list(plane_embed_model.parameters())+
        list(cross_attention_model.parameters()),
        lr=0.001
    )


    epochs = 20

    # Create a DataLoader
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/train_dataset')

    # Filter to only keep
    good_data_indices = [i for i, data in enumerate(dataset) if data[4][-1] == 1]
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
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, _, _, _, _, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

            # 1) Embed the strokes
            # node_features has shape (1, num_strokes, 6)
            # stroke_embed has shape (1, num_strokes, 16)
            node_features = node_features.to(torch.float32).to(device)
            stroke_embed = stroke_embed_model(node_features)

            # 2) Find all possible faces
            face_indices = Models.sketch_arguments.face_aggregate.face_aggregate(node_features)

            # 3) For each face, build the embedding
            # face_embed has shape (1, num_faces, 32)
            face_embed = plane_embed_model(face_indices, stroke_embed)
            
            # 4) Prepare brep_embedding
            if face_features.shape[1] == 0:
                # is empty program
                brep_embedding = torch.zeros(1, 1, 32, device=device)
            else:
                brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
                            edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
                
                brep_graph.to_device(device)
                face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)
                brep_embedding = torch.cat((face_embedding, edge_embedding, vertex_embedding), dim=1)

            # 5) Do cross attention on face_embedding and brep_embedding
            # output has shape (num_faces, 1)
            output = cross_attention_model(face_embed, brep_embedding)








#---------------------------------- Public Functions ----------------------------------#

train()
