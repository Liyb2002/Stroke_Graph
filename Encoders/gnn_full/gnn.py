import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv
from torch_geometric.data import HeteroData

import Encoders.gnn_full.basic


class SemanticModule(nn.Module):
    def __init__(self, in_channels=6, mlp_channels=16):
        super(SemanticModule, self).__init__()
        self.local_head = Encoders.gnn_full.basic.GeneralHeteroConv(['intersects_mean', 'temp_previous_add',  'represented_by_mean', 'brepcoplanar_max', 'strokecoplanar_max'], in_channels, 32)

        self.layers = nn.ModuleList([
            Encoders.gnn_full.basic.ResidualGeneralHeteroConvBlock(['intersects_mean','temp_previous_add',  'represented_by_mean', 'brepcoplanar_max', 'strokecoplanar_max'], 32, 32),
            Encoders.gnn_full.basic.ResidualGeneralHeteroConvBlock(['intersects_mean','temp_previous_add',  'represented_by_mean', 'brepcoplanar_max', 'strokecoplanar_max'], 32, 32),
            Encoders.gnn_full.basic.ResidualGeneralHeteroConvBlock(['intersects_mean','temp_previous_add',  'represented_by_mean', 'brepcoplanar_max', 'strokecoplanar_max'], 32, 32),
            Encoders.gnn_full.basic.ResidualGeneralHeteroConvBlock(['intersects_mean','temp_previous_add','represented_by_mean', 'brepcoplanar_max', 'strokecoplanar_max'], 32, 32),

        ])


    def forward(self, x_dict, edge_index_dict):

        x_dict = self.local_head(x_dict, edge_index_dict)

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict



class sketch_prediction(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=128):
        super(sketch_prediction, self).__init__()

        self.local_head = nn.Linear(32, 64) 

        self.encoder = SemanticModule()
        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x_dict):
        features = self.local_head(x_dict['brep'])
        return torch.sigmoid(self.decoder(features))


class empty_brep_prediction(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=128):
        super(empty_brep_prediction, self).__init__()

        self.local_head = nn.Linear(32, 64) 

        self.encoder = SemanticModule()
        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x_dict):
        features = self.local_head(x_dict['stroke'])
        return torch.sigmoid(self.decoder(features))
