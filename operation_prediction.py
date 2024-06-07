import Preprocessing.dataloader
import Preprocessing.gnn_graph

import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network

import Encoders.gnn.gnn
import Encoders.program_encoder.program_encoder

import Models.operation_model

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 
from config import device
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural networks
graph_embedding_model = Encoders.gnn.gnn.SemanticModule()
program_embedding_model = Encoders.program_encoder.program_encoder.ProgramEncoder()
SBGCN_model = Preprocessing.SBGCN.SBGCN_network.FaceEdgeVertexGCN()
cross_attention_model = Models.operation_model.CrossAttentionTransformer()

graph_embedding_model.to(device)
program_embedding_model.to(device)
SBGCN_model.to(device)
cross_attention_model.to(device)

# Define training 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(graph_embedding_model.parameters()) + 
                             list(program_embedding_model.parameters()) + 
                             list(SBGCN_model.parameters()) +
                             list(cross_attention_model.parameters()), lr=0.001)

epochs = 10

# Create a DataLoader
dataset = Preprocessing.dataloader.Program_Graph_Dataset()

# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

for epoch in range(epochs):
    # Training
    graph_embedding_model.train()
    program_embedding_model.train()
    SBGCN_model.train()
    cross_attention_model.train()
    
    total_train_loss = 0.0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        node_features, operations_matrix, intersection_matrix, program, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

        # to device 
        node_features = node_features.to(torch.float32).to(device)
        operations_matrix = operations_matrix.to(torch.float32).to(device)
        intersection_matrix = intersection_matrix.to(torch.float32).to(device)

        # graph embedding
        gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix)
        gnn_graph.to_device(device)
        graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)

        # program embedding + brep embedding
        gt_next_token = program[0][-1]
        current_program= program[0][:-1]

        if len(current_program) == 0:
            # is empty program
            program_encoding = torch.zeros(1, 1, 32, device=device)
            brep_embedding = torch.zeros(1, 1, 32, device=device)
        else:
            # is not empty program 
            # program embedding
            program_encoding = program_embedding_model(current_program)
            program_encoding = program_encoding.unsqueeze(0)
            
            # brep embedding
            brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
                        edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
            brep_graph.to_device(device)
            face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)
            brep_embedding = torch.cat((face_embedding, edge_embedding, vertex_embedding), dim=1)

        # Forward pass through cross attention model
        output = cross_attention_model(graph_embedding, program_encoding, brep_embedding)
        loss = criterion(output, gt_next_token.unsqueeze(0))

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

    # Validation
    graph_embedding_model.eval()
    program_embedding_model.eval()
    SBGCN_model.eval()
    cross_attention_model.eval()

    total_val_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
            node_features, operations_matrix, intersection_matrix, program, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

            # to device 
            node_features = node_features.to(torch.float32).to(device)
            operations_matrix = operations_matrix.to(torch.float32).to(device)
            intersection_matrix = intersection_matrix.to(torch.float32).to(device)

            # graph embedding
            gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix)
            gnn_graph.to_device(device)
            graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)

            # program embedding + brep embedding
            gt_next_token = program[0][-1]
            current_program= program[0][:-1]

            if len(current_program) == 0:
                # is empty program
                program_encoding = torch.zeros(1, 1, 32, device=device)
                brep_embedding = torch.zeros(1, 1, 32, device=device)
            else:
                # is not empty program 
                # program embedding
                program_encoding = program_embedding_model(current_program)
                program_encoding = program_encoding.unsqueeze(0)
                
                # brep embedding
                brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
                            edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
                brep_graph.to_device(device)
                face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)
                brep_embedding = torch.cat((face_embedding, edge_embedding, vertex_embedding), dim=1)

            # Forward pass through cross attention model
            output = cross_attention_model(graph_embedding, program_encoding, brep_embedding)
            loss = criterion(output, gt_next_token.unsqueeze(0))

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")
