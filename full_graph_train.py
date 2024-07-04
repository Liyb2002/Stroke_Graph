
import Preprocessing.dataloader
import Preprocessing.gnn_graph_full
import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network


from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/train_dataset')

data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

best_val_loss = float('inf')

for epoch in range(1):
    total_train_loss = 0.0
    
    for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{1} - Training"):
        node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, _, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

        if edge_features.shape[1] == 0:
            continue
        # to device 
        node_features = node_features.to(torch.float32).to(device)
        node_features = node_features.squeeze(0)
        operations_matrix = operations_matrix.to(torch.float32).to(device)
        intersection_matrix = intersection_matrix.to(torch.float32).to(device)
        operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)

        # graph embedding
        edge_features = edge_features.to(torch.float32).to(device)
        edge_features = edge_features.squeeze(0)

        gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
        gnn_graph.set_brep_connection(edge_features, face_feature_gnn_list)
        break