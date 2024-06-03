

import json
import numpy as np
import Preprocessing.proc_CAD.helper
import random
import Preprocessing.proc_CAD.random_gen

import os
from Preprocessing.proc_CAD.basic_class import Face, Edge, Vertex

class Brep:
    def __init__(self):
        self.Faces = []
        self.Edges = []
        self.Vertices = []

        self.op = []
        self.idx = 0
        
    
    def init_sketch_op(self):

        axis = np.random.choice(['x', 'y', 'z'])
        points, normal = Preprocessing.proc_CAD.random_gen.generate_random_rectangle(axis)
        
        self._sketch_op(points, normal)


    def _sketch_op(self, points, normal):
        vertex_list = []
        for i, point in enumerate(points):
            vertex_id = f"vertex_{self.idx}_{i}"
            vertex = Vertex(vertex_id, point.tolist())
            self.Vertices.append(vertex)
            vertex_list.append(vertex)

        num_vertices = len(vertex_list)
        for i in range(num_vertices):
            edge_id = f"edge_{self.idx}_{i}"
            edge = Edge(edge_id, [vertex_list[i], vertex_list[(i+1) % num_vertices]])  # Loop back to first vertex to close the shape
            edge.fillet_edge()
            self.Edges.append(edge)

        face_id = f"face_{self.idx}_{0}"
        face = Face(face_id, vertex_list, normal)
        self.Faces.append(face)
        
        self.idx += 1
        self.op.append(['sketch'])


    def regular_sketch_op(self):

        faces_with_future_sketch = [face for face in self.Faces if face.future_sketch ]
        if not faces_with_future_sketch:
            return False
        target_face = random.choice(faces_with_future_sketch)

        boundary_points = [vert.position for vert in target_face.vertices]
        normal = [ 0 - normal for normal in target_face.normal]

        # cases = ['create_circle', 'find_rectangle', 'find_triangle', 'triangle_to_cut']
        cases = ['find_rectangle', 'find_triangle', 'triangle_to_cut']
        selected_case = random.choice(cases)
        if selected_case == 'create_circle':
            radius = Preprocessing.proc_CAD.random_gen.generate_random_cylinder_radius()
            center = Preprocessing.proc_CAD.helper.random_circle(boundary_points, normal)
            self.regular_sketch_circle(normal, radius, center)
            return 

        if selected_case == 'find_rectangle':
            random_polygon_points = Preprocessing.proc_CAD.helper.find_rectangle_on_plane(boundary_points, normal)

        if selected_case == 'find_triangle':
            random_polygon_points = Preprocessing.proc_CAD.helper.find_triangle_on_plane(boundary_points, normal)

        if selected_case == 'triangle_to_cut':
            random_polygon_points = Preprocessing.proc_CAD.helper.find_triangle_to_cut(boundary_points, normal)

        self._sketch_op(random_polygon_points, normal)


    def regular_sketch_circle(self, normal, radius, center):
        face_id = f"face_{self.idx}_{0}"
        face = Face(face_id, [], normal)
        face.circle(radius, center)
        self.Faces.append(face)
        
        self.idx += 1
        self.op.append(['sketch'])


    def add_extrude_add_op(self):
        amount = Preprocessing.proc_CAD.random_gen.generate_random_extrude_add()

        sketch_face = self.Faces[-1]
        sketch_face_opposite_normal = [-x for x in sketch_face.normal]

        new_vertices = []
        new_edges = []
        new_faces = []

        for i, vertex in enumerate(sketch_face.vertices):

            new_pos = [vertex.position[j] + sketch_face_opposite_normal[j] * amount for j in range(3)]
            vertex_id = f"vertex_{self.idx}_{i}"
            new_vertex = Vertex(vertex_id, new_pos)
            self.Vertices.append(new_vertex)
            new_vertices.append(new_vertex)

        num_vertices = len(new_vertices)
        for i in range(num_vertices):
            edge_id = f"edge_{self.idx}_{i}"
            edge = Edge(edge_id, [new_vertices[i], new_vertices[(i+1) % num_vertices]])  # Loop back to first vertex to close the shape
            self.Edges.append(edge)
            new_edges.append(edge)

        face_id = f"face_{self.idx}_{0}"
        new_face = Face(face_id, new_vertices, sketch_face_opposite_normal)
        self.Faces.append(new_face)
        new_faces.append(new_face)
        
        
        #create side edges and faces
        for i in range(num_vertices):
            # Vertical edges from old vertices to new vertices
            vertical_edge_id = f"edge_{self.idx}_{i+num_vertices}"
            vertical_edge = Edge(vertical_edge_id, [sketch_face.vertices[i], new_vertices[i]])
            self.Edges.append(vertical_edge)

            # Side faces formed between pairs of old and new vertices
            side_face_id = f"face_{self.idx}_{i}"
            side_face_vertices = [
                sketch_face.vertices[i], new_vertices[i],
                new_vertices[(i + 1) % num_vertices], sketch_face.vertices[(i + 1) % num_vertices]
            ]
            normal = Preprocessing.proc_CAD.helper.compute_normal(side_face_vertices, new_vertices[(i + 2) % num_vertices])
            side_face = Face(side_face_id, side_face_vertices, normal)
            self.Faces.append(side_face)

        self.idx += 1
        self.op.append(['extrude_addition', sketch_face.id, amount])

    def add_extrude_subtract_op(self):
        amount = Preprocessing.proc_CAD.random_gen.generate_random_extrude_subtract()

        sketch_face = self.Faces[-1]
        sketch_face_opposite_normal = [-x for x in sketch_face.normal]

        new_vertices = []
        new_edges = []
        new_faces = []

        for i, vertex in enumerate(sketch_face.vertices):

            new_pos = [vertex.position[j] - sketch_face_opposite_normal[j] * abs(amount) for j in range(3)]
            vertex_id = f"vertex_{self.idx}_{i}"
            new_vertex = Vertex(vertex_id, new_pos)
            self.Vertices.append(new_vertex)
            new_vertices.append(new_vertex)

        num_vertices = len(new_vertices)
        for i in range(num_vertices):
            edge_id = f"edge_{self.idx}_{i}"
            edge = Edge(edge_id, [new_vertices[i], new_vertices[(i+1) % num_vertices]])  # Loop back to first vertex to close the shape
            self.Edges.append(edge)
            new_edges.append(edge)

        face_id = f"face_{self.idx}_{0}"
        new_face = Face(face_id, new_vertices, sketch_face_opposite_normal)
        self.Faces.append(new_face)
        new_faces.append(new_face)
        
        
        #create side edges and faces
        for i in range(num_vertices):
            # Vertical edges from old vertices to new vertices
            vertical_edge_id = f"edge_{self.idx}_{i+num_vertices}"
            vertical_edge = Edge(vertical_edge_id, [sketch_face.vertices[i], new_vertices[i]])
            self.Edges.append(vertical_edge)

            # Side faces formed between pairs of old and new vertices
            side_face_id = f"face_{self.idx}_{i}"
            side_face_vertices = [
                sketch_face.vertices[i], new_vertices[i],
                new_vertices[(i + 1) % num_vertices], sketch_face.vertices[(i + 1) % num_vertices]
            ]
            normal = Preprocessing.proc_CAD.helper.compute_normal(side_face_vertices, new_vertices[(i + 2) % num_vertices])
            side_face = Face(side_face_id, side_face_vertices, normal)
            self.Faces.append(side_face)

        self.idx += 1
        self.op.append(['extrude_subtraction', sketch_face.id, amount])

    def random_fillet(self):
        
        edge_with_round = [edge for edge in self.Edges if not edge.round]
        if not edge_with_round:
            return False
        target_edge = random.choice(edge_with_round)

        amount = Preprocessing.proc_CAD.random_gen.generate_random_fillet()
        target_edge.fillet_edge()

        verts_pos = []
        verts_id = []
        new_vert_pos = []

        for vert in target_edge.vertices:
            verts_pos.append(vert.position)
            verts_id.append(vert.id)
            neighbor_verts = Preprocessing.proc_CAD.helper.get_neighbor_verts(vert,target_edge,  self.Edges)
            new_vert_pos.append(Preprocessing.proc_CAD.helper.compute_fillet_new_vert(vert, neighbor_verts, amount))
        
        new_A = new_vert_pos[0][0]
        new_B = new_vert_pos[0][1]
        new_C = new_vert_pos[1][0]
        new_D = new_vert_pos[1][1]


        #move old vertex to new_A and new_C
        moved_verts_pos = [new_A, new_C]
        
        #create 2 new verts from new_B and new_D
        new_vert_B = Vertex(f"vertex_{self.idx}_0", new_B)
        new_vert_D = Vertex(f"vertex_{self.idx}_1", new_D)
        self.Vertices.append(new_vert_B)
        self.Vertices.append(new_vert_D)


        #create edge that connect new_B and new_D
        new_edge_id = f"edge_{self.idx}_0"
        new_edge = Edge(new_edge_id, [new_vert_B, new_vert_D])
        self.Edges.append(new_edge)

        #need to change the edge connecting neighbor_verts[0] - old_vert to neighbor_verts[0] - new_vert_B
        edge_vertex_pair = []
        for vert in target_edge.vertices:
            neighbor_verts = Preprocessing.proc_CAD.helper.get_neighbor_verts(vert,target_edge, self.Edges)

            need_to_change_edge = Preprocessing.proc_CAD.helper.find_edge_from_verts(vert, neighbor_verts[1], self.Edges)

            if vert == target_edge.vertices[0]:
                edge_vertex_pair.append([need_to_change_edge.id, neighbor_verts[1].id, new_vert_B.id])

                #connect neighbor_verts[1] with new_vert_B and new_vert_D
                edge1 = Edge(f"edge_{self.idx}_1", [new_vert_B, vert])  
                self.Edges.append(edge1)
            else:
                edge_vertex_pair.append([need_to_change_edge.id, neighbor_verts[1].id, new_vert_D.id])
                
                #connect neighbor_verts[1] with new_vert_B and new_vert_D
                edge2 = Edge(f"edge_{self.idx}_2", [new_vert_D, vert])  
                self.Edges.append(edge2)
            

        self.idx += 1
        self.op.append(['fillet', target_edge.id, 
                        {'amount': amount}, 
                        {'old_verts_pos': verts_pos},
                        {'new_verts_pos': moved_verts_pos},
                        {'verts_id': verts_id},
                        {'need_to_change_edge': edge_vertex_pair},
                        ])

    def write_to_json(self, data_directory = None):
        
        #clean everything in the folder
        folder = os.path.join(os.path.dirname(__file__), 'canvas')
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        if data_directory:
            folder = data_directory

        
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        #start writing program
        filename = os.path.join(folder, 'Program.json')
        data = []
        for count in range(0, self.idx):
            op = self.op[count][0]
            self.write_Op(self.op[count], count, data)
                
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Data saved to {filename}")


    def write_Op(self, Op, index, data):
        operation = {
            'operation': Op,
            'faces': [],
            'edges': [],
            'vertices': []
        }

                # Add each point with an ID to the vertices list
        
        for face in self.Faces:
            if face.id.split('_')[1] == str(index):

                if face.is_cirlce:
                    face = {
                    'id': face.id,
                    'radius': face.radius,
                    'center': [pt for pt in face.center],
                    'normal': [float(n) if isinstance(n, np.floating) else int(n) for n in face.normal]
                    }
                else:
                    face = {
                        'id': face.id,
                        'vertices': [vertex.id for vertex in face.vertices],
                        'normal': [float(n) if isinstance(n, np.floating) else int(n) for n in face.normal]
                    }

                operation['faces'].append(face)

        for edge in self.Edges:
            if edge.id.split('_')[1] == str(index):
                
                edge = {
                    'id': edge.id,
                    'vertices': [vertex.id for vertex in edge.vertices]
                }
                operation['edges'].append(edge)


        
        for vertex in self.Vertices:
            if vertex.id.split('_')[1] == str(index):
                vertex = {
                    'id': vertex.id,
                    'coordinates': vertex.position  # Convert numpy array to list for JSON serialization
                }
                operation['vertices'].append(vertex)
        

        data.append(operation)

        return data
                

