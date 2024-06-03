import Preprocessing.dataloader

from torch.utils.data import DataLoader
from tqdm import tqdm 


dataset = Preprocessing.dataloader.Program_Graph_Dataset()

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
for batch in tqdm(data_loader):
    node_features, operations_matrix, intersection_matrix, program, face_embeddings, edge_embeddings, vertex_embeddings = batch

    print("node_features", node_features.shape)