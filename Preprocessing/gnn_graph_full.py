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

operations_dict = {     "terminate": 0,
                        "sketch": 1,
                        "extrude": 2,
                        "fillet": 3
                    } 

class SketchHeteroData(HeteroData):
    def __init__(self, node_features, operations_matrix, intersection_matrix, operations_order_matrix):
        super(SketchHeteroData, self).__init__()

        # Node features and labels
        self['stroke'].x = node_features
        self['stroke'].y = operations_matrix
        self['stroke'].z = operations_order_matrix
        
        # Order of nodes (sequential order of reading)
        num_strokes = node_features.shape[0]
        order = list(range(num_strokes))
        self['stroke'].order = torch.tensor(order, dtype=torch.long)
        
        # Intersection matrix to edge indices
        intersection_matrix = intersection_matrix.squeeze(0)
        edge_indices = torch.nonzero(intersection_matrix == 1).t()
        self['stroke', 'intersects', 'stroke'].edge_index = edge_indices.long()

        # Temporal edge index (order of nodes)
        temporal_edge_index = [order[:-1], order[1:]]
        temporal_edge_tensor = torch.tensor(temporal_edge_index, dtype=torch.long).contiguous()
        self['stroke', 'temp_previous', 'stroke'].edge_index = temporal_edge_tensor


        self.intersection_matrix = intersection_matrix
        self.stroke_coplanar()

    def set_brep_connection(self, brep_edge_features):
        n = self['stroke'].x.shape[0]
        m = brep_edge_features.shape[0]
        
        # Initialize the (n, 1) matrix with zeros
        connection_matrix = torch.zeros((n, 1), dtype=torch.float32)
        
        # Compare each node feature with each edge feature
        for i in range(n):
            for j in range(m):
                if torch.equal(self['stroke'].x[i], brep_edge_features[j]):
                    connection_matrix[i] = 1
                    break  # No need to check further if a match is found

        # Concatenate the connection_matrix with node_features
        self['stroke'].x = torch.cat((self['stroke'].x, connection_matrix), dim=1)
  
    

    def stroke_coplanar(self):
        node_features = self['stroke'].x

        x_values = set()
        y_values = set()
        z_values = set()

        # Iterate through each line in node_features
        for line in node_features:
            x1, y1, z1, x2, y2, z2 = line
            x_values.update([x1.item(), x2.item()])
            y_values.update([y1.item(), y2.item()])
            z_values.update([z1.item(), z2.item()])

        # Convert sets to sorted lists
        x_values = sorted(list(x_values))
        y_values = sorted(list(y_values))
        z_values = sorted(list(z_values))

        # Initialize lists to store plane indices
        x_planes = []
        y_planes = []
        z_planes = []

        # Iterate through unique values to create plane lists
        for x in x_values:
            plane = []
            for idx, line in enumerate(node_features):
                x1, y1, z1, x2, y2, z2 = line
                if x1.item() == x and x2.item() == x:
                    plane.append(idx)
            if len(plane) >= 3:
                x_planes.append(plane)

        for y in y_values:
            plane = []
            for idx, line in enumerate(node_features):
                x1, y1, z1, x2, y2, z2 = line
                if y1.item() == y and y2.item() == y:
                    plane.append(idx)
            if len(plane) >= 3:
                y_planes.append(plane)

        for z in z_values:
            plane = []
            for idx, line in enumerate(node_features):
                x1, y1, z1, x2, y2, z2 = line
                if z1.item() == z and z2.item() == z:
                    plane.append(idx)
            if len(plane) >= 3:
                z_planes.append(plane)

        planes = x_planes + y_planes + z_planes

        coplanar_matrix = torch.zeros((node_features.shape[0], node_features.shape[0]), dtype=torch.float32)
        for sublist in planes:
            for i in range(len(sublist)):
                for j in range(i + 1, len(sublist)):
                    coplanar_matrix[sublist[i], sublist[j]] = 1
                    coplanar_matrix[sublist[j], sublist[i]] = 1

        edge_indices = torch.nonzero(coplanar_matrix == 1).t()
        self['stroke', 'strokecoplanar', 'stroke'].edge_index = edge_indices.long()

        return coplanar_matrix




def build_graph(stroke_dict):
    num_strokes = len(stroke_dict)
    num_operations = len(operations_dict)
    num_operation_counts = 0

    # find the total number of operations
    for i, (_, stroke) in enumerate(stroke_dict.items()):
        for index in stroke.Op_orders:
            if index > num_operation_counts:
                num_operation_counts = index

    # a map that maps stroke_id (e.g 'edge_0_0' to 0)
    stroke_id_to_index = {}



    node_features = np.zeros((num_strokes, 6))
    operations_matrix = np.zeros((num_strokes, num_operations))
    intersection_matrix = np.zeros((num_strokes, num_strokes))
    operations_order_matrix = np.zeros((num_strokes, num_operation_counts+1))


    for i, (_, stroke) in enumerate(stroke_dict.items()):

        # build node_features
        # node_features has shape num_strokes x 6, which is the starting and ending point
        start_point = stroke.vertices[0].position
        end_point = stroke.vertices[1].position
        node_features[i, :3] = start_point
        node_features[i, 3:] = end_point

        # build operations_matrix
        # operations_matrix has shape num_strokes x num_type_ops
        for op in stroke.Op:
            if op in operations_dict:
                op_index = operations_dict[op]
                operations_matrix[i, op_index] = 1
        
        stroke_id_to_index = {stroke_id: index for index, stroke_id in enumerate(stroke_dict)}

        # build intersection_matrix
        # intersection_matrix has shape num_strokes x num_strokes
        for connected_id in stroke.connected_edges:
            if connected_id in stroke_id_to_index:
                j = stroke_id_to_index[connected_id]
                intersection_matrix[i, j] = 1
                intersection_matrix[j, i] = 1

        # build operation_order_matrix
        # operation_order_matrix has shape num_strokes x num_ops
        for stroke_op_count in stroke.Op_orders:
            operations_order_matrix[i, stroke_op_count] = 1


    # graph = SketchHeteroData(node_features, operations_matrix, intersection_matrix)
    # graph.output_info()

    return node_features, operations_matrix, intersection_matrix, operations_order_matrix

