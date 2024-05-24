import torch
from torch_geometric.data import Data, HeteroData


class GraphHeteroData(HeteroData):
    def __init__(self, face_features, edge_features, vertex_features, 
                 edge_index_face_edge, edge_index_edge_vertex, edge_index_face_face_list,
                 index_id, index_counter):
        super(GraphHeteroData, self).__init__()


        self['face'].x = self.preprocess_features(face_features)
        self['edge'].x = self.preprocess_features(edge_features)
        self['vertex'].x = self.preprocess_features(vertex_features)
        self['face'].y = index_id

        self['face'].num_nodes = len(face_features)
        self['edge'].num_nodes = len(edge_features)
        self['vertex'].num_nodes = len(vertex_features)

        self['face', 'connects', 'edge'].edge_index = edge_index_face_edge
        self['edge', 'connects', 'vertex'].edge_index = edge_index_edge_vertex
        self['edge', 'connects', 'face'].edge_index = self.reverse_edge(edge_index_face_edge)
        self['vertex', 'connects', 'edge'].edge_index = self.reverse_edge(edge_index_edge_vertex)
        self['face', 'connects', 'face'].edge_index = edge_index_face_face_list

        self['face'].z = self.build_adjacency_matrix(
            edge_index_face_edge, edge_index_edge_vertex, edge_index_face_face_list,
            index_counter)

    def to_device(self, device):
        for key, value in self.items():
            if torch.is_tensor(value):
                self[key] = value.to(device)

    def count_nodes(self):
        num_faces = len(self['face'].x)
        num_edges = len(self['edge'].x)
        num_vertices = len(self['vertex'].x)
        
        print("Number of faces:", num_faces)
        print("Number of edges:", num_edges)
        print("Number of vertices:", num_vertices)

    def preprocess_features(self, features):
        processed_features = [] 
        for _, f in features:
            processed_features.append(f)
        
        return torch.tensor(processed_features)

    def reverse_edge(self, edge_list):
        reversed_lst = []
        for sublist in edge_list:
            reversed_lst.append([sublist[1], sublist[0]])
        return reversed_lst


    def build_adjacency_matrix(self, edge_index_face_edge, edge_index_edge_vertex, edge_index_face_face_list, num_nodes):
        adjacency_matrix = torch.zeros(num_nodes, num_nodes)
        
        for edge in edge_index_face_face_list:
            face1_id, face2_id = edge
            adjacency_matrix[face1_id, face2_id] = 1
            adjacency_matrix[face2_id, face1_id] = 1 
        
        for edge in edge_index_face_edge:
            face_id, edge_id = edge
            adjacency_matrix[face_id, edge_id] = 1
            adjacency_matrix[edge_id, face_id] = 1  
        
        for edge in edge_index_edge_vertex:
            edge_id, vertex_id = edge
            adjacency_matrix[edge_id, vertex_id] = 1
            adjacency_matrix[vertex_id, edge_id] = 1
        
        return adjacency_matrix
