from torch.utils.data import Dataset
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import pickle

import gnn_graph
import proc_CAD.helper

class Program_Graph_Dataset(Dataset):
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), 'dataset')
        self.data_dirs = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
    
    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx):
        data_dir = self.data_dirs[idx]
        data_path = os.path.join(self.data_path, data_dir)

        # 1) Load graph
        graph_path = os.path.join(data_path, 'stroke_cloud_graph.pkl')
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        # Three matrices to build the graph
        node_features = graph_data['node_features']
        operations_matrix = graph_data['operations_matrix']
        intersection_matrix = graph_data['intersection_matrix']


        # 2) Load Program
        program_file_path = os.path.join(data_path, 'Program.json')
        program = proc_CAD.helper.program_to_string(program_file_path)
        print("program", len(program))
        print("----------")
        return idx



dataset = Program_Graph_Dataset()

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

for batch in tqdm(data_loader):
    idx = batch
