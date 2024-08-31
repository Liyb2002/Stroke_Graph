import json
from Preprocessing.proc_CAD.basic_class import Face, Edge, Vertex
import Preprocessing.proc_CAD.line_utils

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.interpolate import CubicSpline


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
            for index, op in enumerate(data):
                self.parse_op(op, index)
            
        
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


    def vis_stroke_cloud(self, directory, show=False, target_Op=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Remove grid and axes
        ax.grid(False)
        ax.set_axis_off()

        # Initialize min and max limits
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        z_min, z_max = float('inf'), float('-inf')

        for _, edge in self.edges.items():
            # Determine line color, alpha, and thickness based on edge type
            if edge.edge_type == 'feature_line':
                line_color = 'black'
                line_alpha = np.random.uniform(0.55, 0.85)
                line_thickness = np.random.uniform(0.7, 0.9)
            elif edge.edge_type == 'construction_line':
                line_color = 'black'
                line_alpha = np.random.uniform(0.15, 0.3)
                line_thickness = np.random.uniform(0.4, 0.6)

            # Get edge points and perturb them to create a hand-drawn effect
            points = [vertex.position for vertex in edge.vertices]
            if len(points) == 2:
                # Original points
                x_values = np.array([points[0][0], points[1][0]])
                y_values = np.array([points[0][1], points[1][1]])
                z_values = np.array([points[0][2], points[1][2]])

                # Update min and max limits for each axis
                x_min, x_max = min(x_min, x_values.min()), max(x_max, x_values.max())
                y_min, y_max = min(y_min, y_values.min()), max(y_max, y_values.max())
                z_min, z_max = min(z_min, z_values.min()), max(z_max, z_values.max())

                # Add small random perturbations to make the line appear hand-drawn
                perturb_factor = 0.002  # Adjusted perturbation factor
                perturbations = np.random.normal(0, perturb_factor, (10, 3))  # 10 intermediate points

                # Create interpolated points for smoother curves
                t = np.linspace(0, 1, 10)  # Parameter for interpolation
                x_interpolated = np.linspace(x_values[0], x_values[1], 10) + perturbations[:, 0]
                y_interpolated = np.linspace(y_values[0], y_values[1], 10) + perturbations[:, 1]
                z_interpolated = np.linspace(z_values[0], z_values[1], 10) + perturbations[:, 2]

                # Use cubic splines to smooth the perturbed line
                cs_x = CubicSpline(t, x_interpolated)
                cs_y = CubicSpline(t, y_interpolated)
                cs_z = CubicSpline(t, z_interpolated)

                # Smooth curve points
                smooth_t = np.linspace(0, 1, 100)
                smooth_x = cs_x(smooth_t)
                smooth_y = cs_y(smooth_t)
                smooth_z = cs_z(smooth_t)

                # Plot edges with randomized line thickness and alpha
                ax.plot(smooth_x, smooth_y, smooth_z, color=line_color, alpha=line_alpha, linewidth=line_thickness)

        # Compute the center of the shape
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        # Compute the maximum difference across x, y, z directions
        max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

        # Set the same limits for x, y, and z axes centered around the computed center
        ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
        ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
        ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

        if show:
            plt.show()

        filepath = os.path.join(directory, '3d_visualization.png')
        plt.savefig(filepath)
        plt.close(fig)

    

    def parse_op(self, Op, index):
        op = Op['operation'][0]

        if op == 'terminate':
            construction_lines = Preprocessing.proc_CAD.line_utils.whole_bounding_box_lines(self.edges)
            for line in construction_lines:
                line.set_edge_type('construction_line')
                line.set_order_count(self.order_count)
                line.set_Op(op, index)
                self.order_count += 1
                self.edges[line.id] = line

            # self.edges = proc_CAD.line_utils.remove_duplicate_lines(self.edges)
            self.edges = Preprocessing.proc_CAD.line_utils.perturbing_lines(self.edges)
            return

        for vertex_data in Op['vertices']:
            vertex = Vertex(id=vertex_data['id'], position=vertex_data['coordinates'])
            self.vertices[vertex.id] = vertex


        cur_op_vertex_ids = []
        new_edges = []
        for edge_data in Op['edges']:
            vertices = [self.vertices[v_id] for v_id in edge_data['vertices']]

            for v_id in edge_data['vertices']:
                cur_op_vertex_ids.append(v_id)

            edge = Edge(id=edge_data['id'], vertices=vertices)
            edge.set_Op(op, index)
            edge.set_order_count(self.order_count)
            new_edges.append(edge)

            self.order_count += 1
            self.edges[edge.id] = edge

        construction_lines = []
        # Now, we need to generate the construction lines
        if op == 'sketch':
            construction_lines = Preprocessing.proc_CAD.line_utils.midpoint_lines(new_edges)
            construction_lines += Preprocessing.proc_CAD.line_utils.diagonal_lines(new_edges)                

        if op == 'extrude':
            construction_lines = Preprocessing.proc_CAD.line_utils.projection_lines(new_edges)
            construction_lines += Preprocessing.proc_CAD.line_utils.bounding_box_lines(new_edges)
            # construction_lines = Preprocessing.proc_CAD.line_utils.grid_lines(self.edges, new_edges)

        for line in construction_lines:
            line.set_edge_type('construction_line')
            line.set_order_count(self.order_count)
            line.set_Op(op, index)
            self.order_count += 1
            self.edges[line.id] = line
        

        #find the edges that has the current operation 
        #but not created by the current operation
        self.find_unwritten_edges(cur_op_vertex_ids, op, index)

        for face_data in Op['faces']:
            vertices = [self.vertices[v_id] for v_id in face_data['vertices']]
            normal = face_data['normal']
            face = Face(id=face_data['id'], vertices=vertices, normal=normal)
            self.faces[face.id] = face          


    def adj_edges(self):

        def vert_on_line(vertex, edge):
            # Get the two vertices of the edge
            v1, v2 = edge.vertices

            # Get positions of the vertices
            p1 = v1.position
            p2 = v2.position
            p3 = vertex.position

            # Check if the vertex is one of the line endpoints
            if p3 == p1 or p3 == p2:
                return True

            # Compute vectors
            vec1 = (p2[0] - p1[0], p2[1] - p1[1])
            vec2 = (p3[0] - p1[0], p3[1] - p1[1])

            # Check if vectors are collinear by cross product
            cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]

            # If cross product is zero, the vectors are collinear (the vertex is on the line)
            if cross_product != 0:
                return False

            # Check if p3 is between p1 and p2 using the dot product
            dot_product = (p3[0] - p1[0]) * (p2[0] - p1[0]) + (p3[1] - p1[1]) * (p2[1] - p1[1])
            if dot_product < 0:
                return False

            squared_length = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
            if dot_product > squared_length:
                return False

            return True

        for edge_id, edge in self.edges.items():
            connected_edge_ids = set()  

            for vertex in edge.vertices:
                for other_edge_id, other_edge in self.edges.items():
                    if other_edge_id != edge_id and vert_on_line(vertex, other_edge):
                        connected_edge_ids.add(other_edge_id)
            
            edge.connected_edges = list(connected_edge_ids)

            # print(f"Edge {edge_id} is connected to edges: {list(connected_edge_ids)}")


    def find_unwritten_edges(self, cur_op_vertex_ids, op, index):
        vertex_id_set = set(cur_op_vertex_ids)

        for edge_id, edge in self.edges.items():
            if all(vertex.id in vertex_id_set for vertex in edge.vertices):
                edge.set_Op(op, index)

    
    def map_id_to_count(self):
        for edge_id, edge in self.edges.items():
            self.id_to_count[edge_id] = edge.order_count


def run(directory):
    file_path = os.path.join(directory, 'Program.json')

    stroke_cloud_class = create_stroke_cloud(file_path)
    stroke_cloud_class.read_json_file()

    # stroke_cloud_class.vis_stroke_cloud(directory, show = True)

    return stroke_cloud_class.edges, stroke_cloud_class.faces