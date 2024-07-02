import Preprocessing.dataloader
import Models.sketch_model_helper
import Preprocessing.gnn_graph
import stroke_cloud_annotate

from Preprocessing.SBGCN import SBGCN_network
from Encoders.gnn.gnn import SemanticModule
from Models.sketch_model import BrepStrokeCloudAttention_Reverse, BrepStrokeCloudAttention


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
        from Models.sketch_model import BrepStrokeCloudAttention, BrepStrokeCloudAttention_Reverse, StrokeEmbeddingNetwork, PlaneEmbeddingNetwork

        # Initialize annotate models
        self.SBGCN_model = SBGCN_network.FaceEdgeVertexGCN()
        self.graph_embedding_model = SemanticModule()
        self.BrepStrokeCloudAttention = BrepStrokeCloudAttention()
        
        # Move models to device
        self.SBGCN_model.to(device)
        self.graph_embedding_model.to(device)
        self.BrepStrokeCloudAttention.to(device)


        # Initialize stroke embedding networks
        self.stroke_embed_network = StrokeEmbeddingNetwork()
        self.stroke_embed_network.to(device)
        

        # Initialize predict models
        self.BrepStrokeCloudAttention_finding = BrepStrokeCloudAttention_Reverse()
        
        # Move models to device
        self.BrepStrokeCloudAttention_finding.to(device)

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

    def save_brep_finding_model(self):
        torch.save(self.stroke_embed_network.state_dict(), os.path.join(self.save_dir, 'stroke_embed_network.pth'))
        torch.save(self.BrepStrokeCloudAttention_finding.state_dict(), os.path.join(self.save_dir, 'BrepStrokeCloudAttention_finding.pth'))

        print("Saved models.")



    def train_brep_face_finding(self):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            list(self.BrepStrokeCloudAttention_finding.parameters()),
            lr=0.0002
        )

        epochs = 200

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
            self.BrepStrokeCloudAttention_finding.train()

            total_train_loss = 0.0

            # Training loop
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
                full_node_features, operations_matrix, intersection_matrix, operations_order_matrix, face_to_stroke, program, face_boundary_points, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch
                edge_left = stroke_cloud_annotate.annotate(self.SBGCN_model, self.graph_embedding_model, self.BrepStrokeCloudAttention, batch)

                if edge_features.shape[1] == 0:
                    continue

                # 1) Find the left edges
                node_features, operations_matrix, intersection_matrix, operations_order_matrix = Models.sketch_model_helper.edit_stroke_cloud(edge_left, full_node_features, operations_matrix, intersection_matrix, operations_order_matrix)
                # 1) Prepare the stroke cloud embedding
                node_features = node_features.to(torch.float32).to(device)
                stroke_embedding = self.stroke_embed_network(node_features)


                # 3) Prepare brep_edges embedding
                brep_edge_embedding = self.stroke_embed_network(edge_features)
                
                # 4) Cross attention on edge_embedding and stroke cloud
                brep_edge_chosen = self.BrepStrokeCloudAttention_finding(brep_edge_embedding, stroke_embedding)
                
                # 5) Prepare the ground truth
                chosen_edges_matrix = Models.sketch_model_helper.math_all_stroke_edges(node_features, edge_features)
                
                # 6) Train
                # Models.sketch_model_helper.vis_stroke_cloud(full_node_features)
                # Models.sketch_model_helper.vis_stroke_cloud(node_features)
                # Models.sketch_model_helper.vis_gt_face(edge_features, gt_matrix, edge_index_face_edge_list, index_id)
                # Models.sketch_model_helper.vis_stroke_cloud(edge_features)
                # Models.sketch_model_helper.vis_gt_face(node_features, np.argwhere(kth_operation == 1)[0, 0].item() , edge_index_face_edge_list, index_id)
                # Models.sketch_model_helper.vis_gt_strokes(edge_features, chosen_edges_matrix)

                loss = criterion(brep_edge_chosen, chosen_edges_matrix)
                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}")

            # Validation loop
            self.BrepStrokeCloudAttention_finding.eval()

            total_val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                    full_node_features, operations_matrix, intersection_matrix, operations_order_matrix, face_to_stroke, program, face_boundary_points, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch
                    edge_left = stroke_cloud_annotate.annotate(self.SBGCN_model, self.graph_embedding_model, self.BrepStrokeCloudAttention, batch)

                    if edge_features.shape[1] == 0:
                        continue

                    # 1) Find the left edges
                    node_features, operations_matrix, intersection_matrix, operations_order_matrix = Models.sketch_model_helper.edit_stroke_cloud(edge_left, full_node_features, operations_matrix, intersection_matrix, operations_order_matrix)
                    # 1) Prepare the stroke cloud embedding
                    node_features = node_features.to(torch.float32).to(device)
                    stroke_embedding = self.stroke_embed_network(node_features)


                    # 3) Prepare brep_edges embedding
                    brep_edge_embedding = self.stroke_embed_network(edge_features)
                    
                    # 4) Cross attention on edge_embedding and stroke cloud
                    brep_edge_chosen = self.BrepStrokeCloudAttention_finding(brep_edge_embedding, stroke_embedding)
                    
                    # 5) Prepare the ground truth
                    chosen_edges_matrix = Models.sketch_model_helper.chosen_all_edge_id(node_features, edge_index_face_edge_list, index_id, edge_features)
                    
                    # 6) Train
                    # Models.sketch_model_helper.vis_stroke_cloud(full_node_features)
                    # Models.sketch_model_helper.vis_stroke_cloud(node_features)
                    # Models.sketch_model_helper.vis_gt_face(edge_features, gt_matrix, edge_index_face_edge_list, index_id)
                    # Models.sketch_model_helper.vis_stroke_cloud(edge_features)
                    # Models.sketch_model_helper.vis_gt_face(node_features, np.argwhere(kth_operation == 1)[0, 0].item() , edge_index_face_edge_list, index_id)
                    # Models.sketch_model_helper.vis_gt_strokes(edge_features, chosen_edges_matrix)

                    val_loss = criterion(brep_edge_chosen, chosen_edges_matrix)
                    total_train_loss += loss.item()

                    # 6) Calculate validation loss
                    total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss  
                    self.save_brep_finding_model()



#---------------------------------- Testing Functions ----------------------------------#
face_finder = FindBrepFace()