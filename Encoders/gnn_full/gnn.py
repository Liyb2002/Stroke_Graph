import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv
from torch_geometric.data import HeteroData

import Encoders.gnn_full.basic


class SemanticModule(nn.Module):
    def __init__(self, in_channels=6, mlp_channels=128):
        super(SemanticModule, self).__init__()
        self.local_head = Encoders.gnn_full.basic.GeneralHeteroConv(['temp_previous_add', 'intersects_mean', 'represented_by_mean', 'brepcoplanar_max'], in_channels, 64)

        self.layers = nn.ModuleList([
            Encoders.gnn_full.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean', 'represented_by_mean', 'brepcoplanar_max'], 64, 128),
            Encoders.gnn_full.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean', 'represented_by_mean', 'brepcoplanar_max'], 128, 128),
            Encoders.gnn_full.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean', 'represented_by_mean', 'brepcoplanar_max'], 128, 128),
            Encoders.gnn_full.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean', 'represented_by_mean', 'brepcoplanar_max'], 128, mlp_channels)
        ])

        self.linear_layer = nn.Linear(mlp_channels, 128) 

    def forward(self, x_dict, edge_index_dict):

        x_dict = self.local_head(x_dict, edge_index_dict)

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)

        x = self.linear_layer(x_dict['brep'])

        return x



class InstanceModule(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=512, mlp_channels = 128):
        super(InstanceModule, self).__init__()

        self.encoder = SemanticModule()
        self.decoder = nn.Sequential(
            nn.Linear(mlp_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 128),
            nn.Linear(128, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        features = self.encoder(x_dict, edge_index_dict)
        return torch.sigmoid(self.decoder(features))
