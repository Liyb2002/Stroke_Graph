from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

from torch.utils.data import Dataset
from itertools import combinations

import torch
import os
from tqdm import tqdm
import Preprocessing.SBGCN.SBGCN_graph

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_edges_3d(edge_features):
    """
    Plot a list of edges in 3D.
    
    Args:
    edge_features (list of list): A list where each element is a list of 6 points representing an edge.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for edge in edge_features:
        # Extract the points for the edge
        x1, y1, z1, x2, y2, z2 = edge
        # Plot the edge
        ax.plot([x1, x2], [y1, y2], [z1, z2], marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

def read_step_file(filename):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)

    if status == 1:  # Check if the read was successful
        step_reader.TransferRoot()  # Transfers the whole STEP file
        shape = step_reader.Shape()  # Retrieves the translated shape
        return shape
    else:
        raise Exception("Error reading STEP file.")

def create_face_node(face):
    vertices = []
    unique_vertices = set()  # To store unique vertices
    
    vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        vertex_coords = BRep_Tool.Pnt(vertex)
        vertex_tuple = (vertex_coords.X(), vertex_coords.Y(), vertex_coords.Z())
        
        if vertex_tuple not in unique_vertices:
            unique_vertices.add(vertex_tuple)
            vertices.append([vertex_coords.X(), vertex_coords.Y(), vertex_coords.Z()])
        
        vertex_explorer.Next()
    
    if len(vertices) == 3:
        vertices.append(vertices[0])
    elif len(vertices) > 4:
        vertices = vertices[:4] 

    # Flatten the list of vertices
    flattened_vertices = [coord for vertex in vertices for coord in vertex]
    return flattened_vertices


def create_face_node_gnn(face):
    face_feature_gnn = []
    edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        edge_features = create_edge_node(edge)
        face_feature_gnn.append(edge_features)
        edge_explorer.Next()

    return face_feature_gnn


    
def create_edge_node(edge):
    properties = GProp_GProps()
    brepgprop.LinearProperties(edge, properties)
    length = properties.Mass()

    vertices = []
    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        vertex_coords = BRep_Tool.Pnt(vertex)
        vertices.append([vertex_coords.X(), vertex_coords.Y(), vertex_coords.Z()])
        vertex_explorer.Next()

    return [vertices[0][0], vertices[0][1], vertices[0][2], vertices[1][0], vertices[1][1], vertices[1][2]]

def create_vertex_node(vertex):
    pt = BRep_Tool.Pnt(vertex)
    return [pt.X(), pt.Y(), pt.Z()]


def check_duplicate(new_feature, feature_list):
    for existing_feature in feature_list:
        if existing_feature == new_feature:
            return 1
    
    return -1


def build_face_to_face(edge_index_face_edge_list):
    edge_to_faces = {}
    for face_id, edge_id in edge_index_face_edge_list:
        if edge_id not in edge_to_faces:
            edge_to_faces[edge_id] = set()
        edge_to_faces[edge_id].add(face_id)
    
    shared_face_pairs = []
    for edge_id, face_ids in edge_to_faces.items():
        if len(face_ids) > 1:
            face_pairs = combinations(face_ids, 2)
            for face_pair in face_pairs:
                shared_face_pairs.append(sorted(face_pair))
    
    shared_face_pairs = [list(pair) for pair in set(tuple(pair) for pair in shared_face_pairs)]
    return shared_face_pairs

def count_type(index_to_type_dict):
    counts = {'face': 0, 'edge': 0, 'vertex': 0}
    result = []
    for value in index_to_type_dict.values():
        counts[value] += 1
        if value == 'face':
            result.append(counts['face'] - 1)
        elif value == 'edge':
            result.append(counts['edge'] - 1)
        elif value == 'vertex':
            result.append(counts['vertex'] - 1)
    return result

def create_graph_from_step_file(step_path):
    shape = read_step_file(step_path)

    edge_features_list = []

    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())

        # Explore edges of the face
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            edge_features = create_edge_node(edge)

            edge_duplicate_id = check_duplicate(edge_features, edge_features_list)
            if edge_duplicate_id != -1:
                edge_explorer.Next()
                continue
            
            edge_features_list.append(edge_features)
            
            edge_explorer.Next()
        
        
        face_explorer.Next()

    
    return edge_features_list


class BRep_Dataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        step_path = self.data_paths[idx]

        return step_path
