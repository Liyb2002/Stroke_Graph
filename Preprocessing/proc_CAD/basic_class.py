
import numpy as np


class Face:
    def __init__(self, id, vertices, normal):
        # print(f"An Face is created with ID: {id}")
        self.id = id
        self.vertices = vertices
        self.normal = normal

        self.future_sketch = True
        self.is_cirlce = False

        self.sketch_plane()
    
    def face_fixed(self):
        self.future_sketch = False

    def circle(self, radius, center):
        self.is_cirlce = True
        self.radius = radius
        self.center = center

    def sketch_plane(self):
        # Ensure the sketch is on a plane
        verts = [vert.position for vert in self.vertices]
        
        x_values = [v[0] for v in verts]
        y_values = [v[1] for v in verts]
        z_values = [v[2] for v in verts]
        
        if len(set(x_values)) == 1:
            self.plane = ('x', round(x_values[0], 4))
            return
        elif len(set(y_values)) == 1:
            self.plane = ('y', round(y_values[0], 4))
            return
        elif len(set(z_values)) == 1:
            self.plane = ('z', round(z_values[0], 4))
            return
        
        self.future_sketch = False
        self.plane = ('None', 0)
    

class Edge:
    def __init__(self, id, vertices):
        # print(f"An edge is created with ID: {id}")
        self.id = id
        self.vertices = vertices
        self.round = False

        self.edge_type = 'maybe_feature_line'

        self.Op = []
        self.Op_orders = []
        self.order_count = 0
        self.connected_edges = []
    
    def fillet_edge(self):
        self.round = True
    
    def set_Op(self, Op, order_count):
        self.Op.append(Op)
        self.Op_orders.append(order_count)
    
    def set_order_count(self, order_count):
        self.order_count = order_count

    def connected_edges(self, edge_id):
        self.connected_edges.append(edge_id)
    
    def set_edge_type(self, new_edge_type):
        self.edge_type = new_edge_type
    
    def set_alpha_value(self):
        if self.edge_type == 'feature_line':
            self.alpha_value = np.random.uniform(0.6, 0.9)
        if self.edge_type == 'construction_line':
            self.alpha_value = np.random.uniform(0.2, 0.3)




class Vertex:
    def __init__(self, id, position):
        # print(f"A vertex is created with ID: {id}")
        self.id = id
        self.position = position


