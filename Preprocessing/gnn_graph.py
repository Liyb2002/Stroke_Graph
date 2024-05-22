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
                        "fillet_line": 0,
                        "extrude_line": 1,
                        "sketch": 2,
                        "feature_line": 3,
                        "silhouette_line": 4,
                        "grid_lines": 5,
                        "section_lines": 6,
                        "circle_square_line": 7,
                        "outline": 8
                    } # taken from line_rendering.py file in CAD2Sketch

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


def build_stroke_to_labels_map(final_edges_json):
    stroke_to_labels = {}
    for i, key in enumerate(final_edges_json.keys()):
        stroke = final_edges_json[key]
        labels = [stroke["feature_id"]]
        
        for original_label in stroke["original_labels"]:
            labels.append(original_label["feature_id"])
        
        labels = list(set(labels))
        
        stroke_to_labels[i] = labels
    
    return stroke_to_labels


def build_stroke_to_operation_map(final_edges_json):
    num_strokes = len(final_edges_json)
    num_operations = len(operations_dict)
    stroke_to_operation = torch.zeros((num_strokes, num_operations))

    for i, key in enumerate(final_edges_json.keys()):
        stroke = final_edges_json[key]
        operations = [stroke["type"]]

        for original_label in stroke["original_labels"]:
            operations.append(original_label["type"])
        
        operations = set(operations)
        for operation in operations:
            op_index = operations_dict[operation]  
            stroke_to_operation[i, op_index] = 1  
    
    return stroke_to_operation


def create_graph_from_json(final_edges_path, parsed_features_path, stroke_dict_path):

    final_edges_json = preprocessing.io_utils.read_json_file(final_edges_path)
    parsed_features_json = preprocessing.io_utils.read_json_file(parsed_features_path)
    stroke_dict_json = preprocessing.io_utils.read_json_file(stroke_dict_path)

    #parse parsed_features
    parsed_features_sequence = {}
    if parsed_features_json is not None:
        for dict_i in parsed_features_json['sequence']:
            parsed_features_sequence[dict_i['index']] = {'type': dict_i['type'],
                                        'node_indices': dict_i['index']} 

    #parse final_edges
    sketch_graph = None

    num_strokes = len(final_edges_json.keys())
    node_features = []
    node_labels = []
    connectivity_matrix = torch.zeros((num_strokes, num_strokes))

    #go through all the strokes
    ordered_stroke_ids = []
    for key in final_edges_json.keys():
        stroke = final_edges_json[key]
        geometry = stroke["geometry"]
        id = stroke["id"]
        ordered_stroke_ids.append(id)
        node_features.append(geometry[0] + geometry[len(geometry)-1])

        node_label = stroke["feature_id"]
        node_labels.append(node_label)

    #each node has feature as its start and ending point
    #each node has label as the feature_id

    #node_features has shape n_strokes x 6
    node_features = torch.tensor(node_features, dtype=torch.float32)

    #node_labels has shape n_strokes x _
    node_labels = torch.tensor(node_labels, dtype=torch.float32)


    #build connectivity matrix
    #connectivity_matrix has shape n_strokes x n_strokes
    stroke_id_set = set(ordered_stroke_ids)
    id_to_index = {stroke_id: index for index, stroke_id in enumerate(ordered_stroke_ids)}
    for stroke_dict_item in stroke_dict_json:
        if stroke_dict_item['id'] in stroke_id_set:
            stroke_index = id_to_index[stroke_dict_item['id']]
            for sublist in stroke_dict_item['intersections']:
                for intersected_id in sublist:
                    if intersected_id in stroke_id_set:
                        intersected_index = id_to_index[intersected_id]
                        connectivity_matrix[stroke_index, intersected_index] = 1
                        connectivity_matrix[intersected_index, stroke_index] = 1

    #build grouping matrix
    #grouping_matrix has shape n_strokes x n_strokes
    # stroke_to_labels = build_stroke_to_labels_map(final_edges_json)

    # grouping_matrix = torch.zeros((num_strokes, num_strokes))
    # for i in range(num_strokes):
    #     labels_i = stroke_to_labels[i]

    #     for j in range(i, num_strokes):
    #         labels_j = stroke_to_labels[j]
    #         if any(label in labels_j for label in labels_i):
    #             grouping_matrix[i, j] = 1
    #             grouping_matrix[j, i] = 1



    #build stroke_to_operation matrix
    #stroke_to_operation matrix has shape n_strokes x n_operations
    stroke_to_operation = build_stroke_to_operation_map(final_edges_json)


    #build temporal_edge_index
    temporal_edge_index = []
    for i in range (num_strokes -1):
        temporal_edge_index.append([i+1, i])

    #build stroke graph
    sketch_graph = SketchHeteroData(node_features, node_labels, stroke_to_operation, connectivity_matrix, temporal_edge_index)

    return sketch_graph



def build_graph(stroke_dict):
    print("\nEdges:")
    for edge_id, edge in stroke_dict.items():
        vertex_ids = [vertex.id for vertex in edge.vertices]
        # Adding checks if 'Op' and 'order_count' are attributes of edge
        ops = getattr(edge, 'Op', 'No operations')
        order_count = getattr(edge, 'order_count', 'No order count')
        connected_edge_ids = getattr(edge, 'connected_edges', None)
    
        print(f"Edge ID: {edge_id}, Vertices: {vertex_ids},  Operations: {ops}, Order Count: {order_count}, Connected Edges: {connected_edge_ids}")


def build_graph_example():
    stroke_dict_path = '/Users/yuanboli/Documents/GitHub/Baseline_Stroke2CAD/dataset/CAD2Sketch/1069/63.86_149.75_1.4/strokes_dict.json'
    parsed_features_path = '/Users/yuanboli/Documents/GitHub/Baseline_Stroke2CAD/dataset/CAD2Sketch/1069/parsed_features.json'
    final_edges_path = '/Users/yuanboli/Documents/GitHub/Baseline_Stroke2CAD/dataset/CAD2Sketch/1069/63.86_149.75_1.4/final_edges.json'


    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    graph = create_graph_from_json(final_edges_path, parsed_features_path, stroke_dict_path)
    graph.to_device(device)



# build_graph_example()