import Preprocessing.dataloader
import Preprocessing.gnn_graph

from torch.utils.data import DataLoader
from tqdm import tqdm 
from config import device
import torch


dataset = Preprocessing.dataloader.Program_Graph_Dataset()

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
for batch in tqdm(data_loader):
    node_features, operations_matrix, intersection_matrix, program, face_embeddings, edge_embeddings, vertex_embeddings = batch

    node_features = node_features.to(torch.float32).to(device)
    operations_matrix = operations_matrix.to(torch.float32).to(device)
    intersection_matrix = intersection_matrix.to(torch.float32).to(device)

    face_embeddings = face_embeddings.to(torch.float32).to(device)
    edge_embeddings = edge_embeddings.to(torch.float32).to(device)
    vertex_embeddings = vertex_embeddings.to(torch.float32).to(device)

    graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix)
    graph.to_device(device)
