import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv
from torch_geometric.data import HeteroData

import gnn.basic

class SemanticModule(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=32, mlp_channels=[64, 32], num_classes = 10):
        super(SemanticModule, self).__init__()
        self.local_head = gnn.basic.GeneralHeteroConv(['temp_previous_add', 'intersects_mean'], in_channels, hidden_channels)


        self.layers = nn.ModuleList([
            gnn.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean'], hidden_channels, hidden_channels),
            gnn.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean'], hidden_channels, hidden_channels),
            gnn.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean'], hidden_channels, hidden_channels),
            gnn.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean'], hidden_channels, mlp_channels[0])
        ])

    def forward(self, x_dict, edge_index_dict):
        
        print("--------------------------------------")
        for key, value in x_dict.items():
            print(f"Key: {key}, Type: {value.dtype}")

        for key, value in edge_index_dict.items():
            print(f"Key: {key}, Type: {value.dtype}")

        x_dict = self.local_head(x_dict, edge_index_dict)

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)

        return x_dict['stroke']


class InstanceModule(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=32, mlp_channels= [64, 32]):
        super(InstanceModule, self).__init__()
        num_classes = 9
        
        self.encoder = SemanticModule(in_channels, hidden_channels, mlp_channels, num_classes)
        self.decoder = nn.Sequential(
            nn.Linear(mlp_channels[0], hidden_channels),  
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_classes)  
        )

    def forward(self, x_dict, edge_index_dict):
        features = self.encoder(x_dict, edge_index_dict)
        return torch.sigmoid(self.decoder(features))  