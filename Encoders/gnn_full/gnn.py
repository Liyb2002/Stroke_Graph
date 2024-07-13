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



class Sketch_brep_prediction(nn.Module):
    def __init__(self, hidden_channels=128):
        super(Sketch_brep_prediction, self).__init__()

        self.local_head = nn.Linear(32, 64) 
        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x_dict):
        features = self.local_head(x_dict['brep'])
        return torch.sigmoid(self.decoder(features))


class Empty_brep_prediction(nn.Module):
    def __init__(self, hidden_channels=128):
        super(Empty_brep_prediction, self).__init__()

        self.local_head = nn.Linear(32, 64) 
        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x_dict):
        features = self.local_head(x_dict['stroke'])
        return torch.sigmoid(self.decoder(features))


class Final_stroke_finding(nn.Module):
    def __init__(self, hidden_channels=128):
        super(Final_stroke_finding, self).__init__()

        self.edge_conv = Encoders.gnn_full.basic.ResidualGeneralHeteroConvBlock(['intersects_mean','temp_previous_add',  'represented_by_mean', 'brepcoplanar_max', 'strokecoplanar_max'], 32, 32)

        self.local_head = nn.Linear(32, 64) 
        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x_dict, edge_index_dict, stroke_weights):
        x_dict['stroke'] = x_dict['stroke'] * stroke_weights
        x_dict = self.edge_conv(x_dict, edge_index_dict)
        features = self.local_head(x_dict['stroke'])
        return torch.sigmoid(self.decoder(features))



class ExtrudingStrokePrediction(nn.Module):
    def __init__(self, in_channels=32, hidden_channels=64):
        super(ExtrudingStrokePrediction, self).__init__()

        self.edge_conv = Encoders.gnn_full.basic.ResidualGeneralHeteroConvBlock(['intersects_mean','temp_previous_add',  'represented_by_mean', 'brepcoplanar_max', 'strokecoplanar_max'], 32, 32)

        self.local_head = nn.Linear(32, 64) 
        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x_dict, edge_index_dict, sketch_strokes_id):

        connected_strokes_mask = torch.zeros_like(sketch_strokes_id, dtype=torch.float32)
        
        for edge in edge_index_dict[('stroke', 'intersects', 'stroke')].t():
            src, dst = edge
            if sketch_strokes_id[src] == 1:
                connected_strokes_mask[dst] = 1
            if sketch_strokes_id[dst] == 1:
                connected_strokes_mask[src] = 1

        x_dict['stroke'] = x_dict['stroke'] + x_dict['stroke'] * (sketch_strokes_id)

        x_dict = self.edge_conv(x_dict, edge_index_dict)
        features = self.local_head(x_dict['stroke'])
        return torch.sigmoid(self.decoder(features))
