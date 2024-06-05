import Preprocessing.dataloader
import Preprocessing.gnn_graph

import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network

import Encoders.gnn.gnn
import Encoders.program_encoder.program_encoder


from torch.utils.data import DataLoader
from tqdm import tqdm 
from config import device
import torch


dataset = Preprocessing.dataloader.Program_Graph_Dataset()

graph_embedding_model = Encoders.gnn.gnn.SemanticModule()
program_embedding_model = Encoders.program_encoder.program_encoder.ProgramEncoder()
SBGCN_model = Preprocessing.SBGCN.SBGCN_network.FaceEdgeVertexGCN()

graph_embedding_model.to(device)
program_embedding_model.to(device)
SBGCN_model.to(device)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
for batch in tqdm(data_loader):
    graph_embedding_model.train()
    node_features, operations_matrix, intersection_matrix, program, face_features_list, edge_features_list, vertex_features_list, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id= batch

    # to device 
    node_features = node_features.to(torch.float32).to(device)
    operations_matrix = operations_matrix.to(torch.float32).to(device)
    intersection_matrix = intersection_matrix.to(torch.float32).to(device)

    # graph embedding
    gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix)
    gnn_graph.to_device(device)
    graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    print("graph_embedding", graph_embedding.shape)

    # program embedding
    program_encoding = program_embedding_model(program)
    print("program_encoding", program_encoding.shape)

    # brep embedding
    brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features_list, edge_features_list, vertex_features_list, 
                 edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
    face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)
    brep_embeddings = torch.cat((face_embedding, edge_embedding, vertex_embedding), dim=1)

    print("Combined embeddings shape:", brep_embeddings.shape)

    print("--------------------")