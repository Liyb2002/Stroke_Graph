import dummy_network
import Preprocessing.dataloader
import stroke_cloud_annotate
import Models.sketch_model_helper

import stroke_cloud_annotate

import os
import torch
from config import device


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


model = dummy_network.CrossAttentionNetwork()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
epochs = 20


dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/train_dataset')
good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 1]
filtered_dataset = Subset(dataset, good_data_indices)
print(f"Total number of sketch data: {len(filtered_dataset)}")

data_loader = DataLoader(filtered_dataset, batch_size=1, shuffle=True)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
        node_features, operations_matrix, intersection_matrix, operations_order_matrix, face_to_stroke, program, face_boundary_points, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

        if edge_features.shape[1] == 0:
            continue
        
        operation_count = len(program[0]) - 1
        boundary_points = face_boundary_points[operation_count]
        boundary_points_matrix = torch.tensor(boundary_points, dtype=torch.float32)

        vertex_features = vertex_features.squeeze(0)

        output = model(vertex_features, boundary_points_matrix)
        gt_matrix = Models.sketch_model_helper.chosen_vertex_id(boundary_points_matrix, vertex_features)
        
        loss = criterion(output, gt_matrix)  
        
        # print("vertex_features", vertex_features)
        # print("boundary_points_matrix",boundary_points_matrix)
        # print("boundary_points_matrix.shape", boundary_points_matrix.shape)
        # print("vertex_features.shape", vertex_features.shape)
        print("gt_matrix", gt_matrix)
        print("output", output)
        print("---------")


        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        epoch_loss += loss.item()
        
    epoch_loss /= len(data_loader.dataset)  # Average loss for epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

