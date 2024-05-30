import json
import proc_CAD.build123.protocol
from proc_CAD.basic_class import Face, Edge, Vertex
import proc_CAD.helper

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class create_stroke_cloud():
    def __init__(self, file_path, output = True):
        self.file_path = file_path

        self.order_count = 0
        self.faces = {}
        self.edges = {}
        self.vertices = {}
        self.id_to_count = {}
        
    def read_json_file(self):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
            for Op in data:
                self.parse_op(Op)
            
        
        self.adj_edges()
        self.map_id_to_count()

        return


    def output(self, onlyStrokes = True):
        print("Outputting details of all components...")

        # Output vertices
        print("\nVertices:")
        if not onlyStrokes:
            for vertex_id, vertex in self.vertices.items():
                print(f"Vertex ID: {vertex_id}, Position: {vertex.position}")

            # Output faces
            print("\nFaces:")
            for face_id, face in self.faces.items():
                vertex_ids = [vertex.id for vertex in face.vertices]
                normal = face.normal
                print(f"Face ID: {face_id}, Vertices: {vertex_ids}, Normal: {normal}")


        # Output edges
        print("\nEdges:")
        for edge_id, edge in self.edges.items():
            vertex_ids = [vertex.id for vertex in edge.vertices]
            # Adding checks if 'Op' and 'order_count' are attributes of edge
            ops = getattr(edge, 'Op', 'No operations')
            order_count = getattr(edge, 'order_count', 'No order count')
            connected_edge_ids = getattr(edge, 'connected_edges', None)
        
            print(f"Edge ID: {edge_id}, Vertices: {vertex_ids},  Operations: {ops}, Order Count: {order_count}, Connected Edges: {connected_edge_ids}")


    def vis_stroke_cloud(self, target_Op = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        
        for _, edge in self.edges.items():

            line_color = 'blue'

            if target_Op is not None and target_Op in edge.Op:
                line_color = 'red'

            
            points = [vertex.position for vertex in edge.vertices]
            if len(points) == 2:
                x_values = [points[0][0], points[1][0]]
                y_values = [points[0][1], points[1][1]]
                z_values = [points[0][2], points[1][2]]
                ax.plot(x_values, y_values, z_values, marker='o', color=line_color)  # Line plot connecting the vertices

        plt.show()

        
    def parse_op(self, Op):
        op = Op['operation'][0]

        if len(Op['faces']) > 0 and 'radius' in Op['faces'][0]:
            print("parse circle")
            return

        for vertex_data in Op['vertices']:
            vertex = Vertex(id=vertex_data['id'], position=vertex_data['coordinates'])
            self.vertices[vertex.id] = vertex


        cur_op_vertex_ids = []
        for edge_data in Op['edges']:
            vertices = [self.vertices[v_id] for v_id in edge_data['vertices']]

            for v_id in edge_data['vertices']:
                cur_op_vertex_ids.append(v_id)

            edge = Edge(id=edge_data['id'], vertices=vertices)
            edge.set_Op(op)
            edge.set_order_count(self.order_count)
            self.order_count += 1
            self.edges[edge.id] = edge
        
        #find the edges that has the current operation 
        #but not created by the current operation
        self.find_unwritten_edges(cur_op_vertex_ids, op)

        for face_data in Op['faces']:
            vertices = [self.vertices[v_id] for v_id in face_data['vertices']]
            normal = face_data['normal']
            face = Face(id=face_data['id'], vertices=vertices, normal=normal)
            self.faces[face.id] = face  

        if op == 'fillet':
            self.parse_fillet(Op)


    def parse_fillet(self, Op):
        verts_ids = Op['operation'][5]['verts_id']

        old_pos = Op['operation'][3]['old_verts_pos']
        new_pos = Op['operation'][4]['new_verts_pos']
        need_to_change_edge = Op['operation'][6]['need_to_change_edge']

        for edge_id, edge in self.edges.items():
            # Get the IDs of the vertices in the current edge
            edge_vertex_ids = [vertex.id for vertex in edge.vertices]
            
            # Check if the two sets are equal
            if set(edge_vertex_ids) == set(verts_ids):
                
                for vertex in edge.vertices:
                    if vertex.position == old_pos[0]:
                        vertex.position = new_pos[0]
                    elif vertex.position == old_pos[1]:
                        vertex.position = new_pos[1]

                edge.set_Op('fillet')

            # We also need to update the edge - vertex in need_to_change_edge
            #write here
            for change in need_to_change_edge:
                edge_id_to_change = change[0]
                vert_id_1 = change[1]
                vert_id_2 = change[2]

                if edge_id == edge_id_to_change:
                    # Find vertices whose IDs match vert_id_1 and vert_id_2
                    vert_1 = None
                    vert_2 = None

                    for vertex_id, vertex in self.vertices.items():
                        if vertex_id == vert_id_1:
                            vert_1 = vertex
                        elif vertex_id == vert_id_2:
                            vert_2 = vertex
                    
                    if vert_1 and vert_2:
                        edge.vertices = [vert_1, vert_2]


        return


    def adj_edges(self):
        for edge_id, edge in self.edges.items():
            connected_edge_ids = set()  

            for vertex in edge.vertices:
                for other_edge_id, other_edge in self.edges.items():
                    if other_edge_id != edge_id and vertex in other_edge.vertices:
                        connected_edge_ids.add(other_edge_id)
            
            edge.connected_edges = list(connected_edge_ids)

            # print(f"Edge {edge_id} is connected to edges: {list(connected_edge_ids)}")


    def find_unwritten_edges(self, cur_op_vertex_ids, op):
        vertex_id_set = set(cur_op_vertex_ids)

        for edge_id, edge in self.edges.items():
            if all(vertex.id in vertex_id_set for vertex in edge.vertices):
                edge.set_Op(op)

    
    def map_id_to_count(self):
        for edge_id, edge in self.edges.items():
            self.id_to_count[edge_id] = edge.order_count
                



# Example usage:

def run(vis = False):
    file_path = os.path.join(os.path.dirname(__file__), 'canvas', 'Program.json')

    stroke_cloud_class = create_stroke_cloud(file_path)
    stroke_cloud_class.read_json_file()
    # parsed_program_class.output()

    if vis:
        stroke_cloud_class.vis_stroke_cloud('sketch')

    return stroke_cloud_class.edges