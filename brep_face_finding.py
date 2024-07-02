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
        

        # Initialize predict models
        self.SBGCN_model_finding = SBGCN_network.FaceEdgeVertexGCN()
        self.graph_embedding_model_finding = SemanticModule()
        self.BrepStrokeCloudAttention_finding = BrepStrokeCloudAttention_Reverse()
        
        # Move models to device
        self.SBGCN_model_finding.to(device)
        self.graph_embedding_model_finding.to(device)
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


    def train_brep_face_finding(self):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            list(self.SBGCN_model_finding.parameters()) +
            list(self.graph_embedding_model_finding.parameters()) +
            list(self.BrepStrokeCloudAttention_finding.parameters()),
            lr=5e-4
        )

        epochs = 20

        dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/train_dataset')
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
            self.SBGCN_model_finding.train()
            self.graph_embedding_model_finding.train()
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
                operations_matrix = operations_matrix.to(torch.float32).to(device)
                intersection_matrix = intersection_matrix.to(torch.float32).to(device)
                operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)

                # graph embedding
                gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
                gnn_graph.to_device(device)
                stroke_cloud_graph_embedding = self.graph_embedding_model_finding(gnn_graph.x_dict, gnn_graph.edge_index_dict)


                # 3) Prepare brep_edges embedding
                brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
                brep_graph.to_device(device)
                brep_face_embedding, brep_edge_embedding, brep_vertex_embedding = self.SBGCN_model_finding(brep_graph)
                # brep_embedding = torch.cat((brep_face_embedding, brep_edge_embedding, brep_vertex_embedding), dim=1)

                # 4) Cross attention on edge_embedding and stroke cloud
                face_chosen = self.BrepStrokeCloudAttention_finding(brep_face_embedding, stroke_cloud_graph_embedding)
                
                # 5) Prepare the ground truth
                gt_matrix = Models.sketch_model_helper.chosen_all_face_id(node_features, edge_index_face_edge_list, index_id, edge_features)
                
                # 6) Train
                # Models.sketch_model_helper.vis_stroke_cloud(full_node_features)
                # Models.sketch_model_helper.vis_stroke_cloud(node_features)
                # Models.sketch_model_helper.vis_gt_face(edge_features, gt_matrix, edge_index_face_edge_list, index_id)
                # Models.sketch_model_helper.vis_stroke_cloud(edge_features)
                # Models.sketch_model_helper.vis_gt_face(node_features, np.argwhere(kth_operation == 1)[0, 0].item() , edge_index_face_edge_list, index_id)
                # Models.sketch_model_helper.vis_gt_strokes(node_features, coplanar_matrix)

                loss = criterion(face_chosen, gt_matrix)
                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}")

            # Validation loop
            self.SBGCN_model_finding.eval()
            self.graph_embedding_model_finding.eval()
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
                    operations_matrix = operations_matrix.to(torch.float32).to(device)
                    intersection_matrix = intersection_matrix.to(torch.float32).to(device)
                    operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)

                    # graph embedding
                    gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
                    gnn_graph.to_device(device)
                    stroke_cloud_graph_embedding = self.graph_embedding_model_finding(gnn_graph.x_dict, gnn_graph.edge_index_dict)


                    # 3) Prepare brep_edges embedding
                    brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
                    brep_graph.to_device(device)
                    brep_face_embedding, brep_edge_embedding, _ = self.SBGCN_model_finding(brep_graph)
                    

                    # 4) Cross attention on edge_embedding and stroke cloud
                    face_chosen = self.BrepStrokeCloudAttention_finding(brep_face_embedding, stroke_cloud_graph_embedding)

                    # 5) Prepare the ground truth
                    gt_matrix = Models.sketch_model_helper.chosen_all_face_id(node_features, edge_index_face_edge_list, index_id, edge_features)

                    # 6) Calculate validation loss
                    val_loss = criterion(face_chosen, gt_matrix)
                    total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")



#---------------------------------- Testing Functions ----------------------------------#
face_finder = FindBrepFace()