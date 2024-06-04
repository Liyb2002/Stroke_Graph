import Preprocessing.dataloader
import Preprocessing.gnn_graph

import Encoders.gnn.gnn
import Encoders.program_encoder.program_encoder

from torch.utils.data import DataLoader
from tqdm import tqdm 
from config import device
import torch


dataset = Preprocessing.dataloader.Program_Graph_Dataset()

graph_embedding_model = Encoders.gnn.gnn.SemanticModule()
program_embedding_model = Encoders.program_encoder.program_encoder.ProgramEncoder()
graph_embedding_model.to(device)
program_embedding_model.to(device)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
for batch in tqdm(data_loader):
    graph_embedding_model.train()
    node_features, operations_matrix, intersection_matrix, program, face_embeddings, edge_embeddings, vertex_embeddings = batch

    # to device 
    node_features = node_features.to(torch.float32).to(device)
    operations_matrix = operations_matrix.to(torch.float32).to(device)
    intersection_matrix = intersection_matrix.to(torch.float32).to(device)
    face_embeddings = face_embeddings.to(torch.float32).to(device)
    edge_embeddings = edge_embeddings.to(torch.float32).to(device)
    vertex_embeddings = vertex_embeddings.to(torch.float32).to(device)

    # graph embedding
    gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix)
    gnn_graph.to_device(device)
    graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)

    #program embedding
    # print("program", program.shape)
    # program_encoding = program_embedding_model(program)

