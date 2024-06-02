from torch.utils.data import Dataset
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import pickle

import gnn_graph
import proc_CAD.helper
import SBGCN.run_SBGCN

class Program_Graph_Dataset(Dataset):
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), 'dataset')
        self.data_dirs = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        self.index_mapping = self._create_index_mapping()

        self.SBGCN_encoder = SBGCN.run_SBGCN.load_pretrained_SBGCN_model()
        print(f"Number of data directories: {len(self.data_dirs)}")
        print(f"Total number of brep_i.step files: {len(self.index_mapping)}")

    def _create_index_mapping(self):
        index_mapping = []
        for data_dir in self.data_dirs:
            canvas_dir_path = os.path.join(self.data_path, data_dir, 'canvas')
            if os.path.exists(canvas_dir_path):
                brep_files = sorted([f for f in os.listdir(canvas_dir_path) if f.startswith('brep_') and f.endswith('.step')])
                for brep_file_path in brep_files:
                    index_mapping.append((data_dir, brep_file_path))
        return index_mapping

    def __len__(self):
        return len(self.index_mapping)


    def __getitem__(self, idx):
        data_dir, brep_file_path = self.index_mapping[idx]
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
        print("num program", len(program))


        # 3) Load Brep
        print("brep_file", brep_file_path)
        return idx
    




dataset = Program_Graph_Dataset()

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
for batch in tqdm(data_loader):
    idx = batch
