import Preprocessing.dataloader
import stroke_cloud_annotate

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

class FindBrepFace():
    def __init__(self):
        current_dir = os.getcwd()
        self.save_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
        
        # Initialize models
        self.SBGCN_model = None
        self.graph_embedding_model = None
        self.BrepStrokeCloudAttention = None

        self.load_annotate_models()
        self.train_brep_face_finding()

    def load_annotate_models(self):
        # Importing models
        from Preprocessing.SBGCN import SBGCN_network
        from Encoders.gnn.gnn import SemanticModule
        from Models.sketch_model import BrepStrokeCloudAttention

        # Initialize models
        self.SBGCN_model = SBGCN_network.FaceEdgeVertexGCN()
        self.graph_embedding_model = SemanticModule()
        self.BrepStrokeCloudAttention = BrepStrokeCloudAttention()
        
        # Move models to device
        self.SBGCN_model.to(device)
        self.graph_embedding_model.to(device)
        self.BrepStrokeCloudAttention.to(device)
        
        # Load models if they exist
        if os.path.exists(os.path.join(self.save_dir, 'SBGCN_model.pth')):
            self.SBGCN_model.load_state_dict(torch.load(os.path.join(self.save_dir, 'SBGCN_model.pth')))
            print("Loaded SBGCN_model")

        if os.path.exists(os.path.join(self.save_dir, 'graph_embedding_model.pth')):
            self.graph_embedding_model.load_state_dict(torch.load(os.path.join(self.save_dir, 'graph_embedding_model.pth')))
            print("Loaded graph_embedding_model")

        if os.path.exists(os.path.join(self.save_dir, 'BrepStrokeCloudAttention.pth')): 
            self.BrepStrokeCloudAttention.load_state_dict(torch.load(os.path.join(self.save_dir, 'BrepStrokeCloudAttention.pth')))
            print("Loaded BrepStrokeCloudAttention")


    def train_brep_face_finding(self):

        epochs = 20

        dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/full_train_dataset')
        good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 1]
        filtered_dataset = Subset(dataset, good_data_indices)
        print(f"Total number of sketch data: {len(filtered_dataset)}")

        # Split dataset into training and validation
        train_size = int(0.8 * len(filtered_dataset))
        val_size = len(filtered_dataset) - train_size
        train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            
            total_train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                node_features, operations_matrix, intersection_matrix, operations_order_matrix, face_to_stroke, program, face_boundary_points, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch
                edge_left = stroke_cloud_annotate.annotate(self.SBGCN_model, self.graph_embedding_model, self.BrepStrokeCloudAttention, batch)
                print("edge_left", edge_left)
        

#---------------------------------- Testing Functions ----------------------------------#
face_finder = FindBrepFace()