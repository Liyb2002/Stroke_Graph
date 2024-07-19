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


class Program_prediction(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=128, num_classes=10, dropout=0.1):
        super(Program_prediction, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.program_encoder = ProgramEncoder()
    
    def forward(self, x_dict, program_tokens):

        if len(program_tokens) == 0:
            program_embedding = torch.zeros(1, 32)
        else:
            program_embedding = self.program_encoder(program_tokens)
        
        attn_output, _ = self.cross_attn(program_embedding, x_dict['stroke'], x_dict['stroke'])
        out = self.norm(program_embedding + attn_output)

        ff_output = self.ff(out)        
        out_mean = ff_output.mean(dim=0)
        
        logits = self.classifier(out_mean)
        return logits
    

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
        x_dict['stroke'] = x_dict['stroke'] + x_dict['stroke'] * (sketch_strokes_id)

        x_dict = self.edge_conv(x_dict, edge_index_dict)
        features = self.local_head(x_dict['stroke'])
        return torch.sigmoid(self.decoder(features))



# ----------------------------------- Other Models ----------------------------------- #

class ProgramEncoder(nn.Module):
    def __init__(self, vocab_size=10, embedding_dim=8, hidden_dim=32):
        super(ProgramEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return lstm_out
