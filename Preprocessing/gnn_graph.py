import os
import torch

import os
import sys
import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from networkx.algorithms import community
import seaborn as sns

operations_dict = {
                        "sketch": 0,
                        "extrude_addition": 1,
                        "extrude_substraction": 2, 
                        "fillet": 3
                    } 

class SketchHeteroData(HeteroData):
    def __init__(self, node_features, node_labels, additional_matrix, connectivity_matrix, temporal_edge_index):
        super(SketchHeteroData, self).__init__()

        self['stroke'].x = node_features
        self['stroke'].y = node_labels
        self['stroke'].z = additional_matrix

        edge_indices = (connectivity_matrix == 1).nonzero(as_tuple=False).t()
        self['stroke', 'intersects', 'stroke'].edge_index = edge_indices

        temporal_edge_tensor = torch.tensor(temporal_edge_index, dtype=torch.long).t().contiguous()
        self['stroke', 'temp_previous', 'stroke'].edge_index = temporal_edge_tensor

        self.connectivity_matrix = connectivity_matrix

    def to_device(self, device):
        for key, value in self.items():
            if torch.is_tensor(value):
                self[key] = value.to(device)






def build_graph(stroke_dict):
    num_strokes = len(stroke_dict)
    num_operations = len(operations_dict)



    node_features = np.zeros((num_strokes, 6))
    operations_matrix = np.zeros((num_strokes, num_operations))


    for i, (_, stroke) in enumerate(stroke_dict.items()):

        # build node_features
        # node_features has shape num_strokes x 6, which is the starting and ending point
        start_point = stroke.vertices[0].position
        end_point = stroke.vertices[1].position
        node_features[i, :3] = start_point
        node_features[i, 3:] = end_point

        # build operations_matrix
        # operations_matrix has shape num_strokes x num_ops
        for op in stroke.Op:
            if op in operations_dict:
                op_index = operations_dict[op]
                operations_matrix[i, op_index] = 1

    
    print("operations_matrix", operations_matrix)
    
    
    return node_features

