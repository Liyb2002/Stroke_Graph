import json
import os
from copy import deepcopy

import proc_CAD.build123.protocol
from proc_CAD.basic_class import Face, Edge, Vertex
import proc_CAD.helper


class parsed_program():
    def __init__(self, file_path, data_directory, output = True):
        self.file_path = file_path
        self.data_directory = data_directory

        canvas_directory = os.path.join(data_directory, 'canvas')
        os.makedirs(canvas_directory, exist_ok=True)

        self.canvas = None
        self.prev_sketch = None
        self.Op_idx = 0
        self.output = output
        
    def read_json_file(self):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
            self.len_program = len(data)
            for i in range(len(data)):
                Op = data[i]
                operation = Op['operation']
                
                if operation[0] == 'sketch':
                    self.parse_sketch(Op)
                
                if operation[0] == 'extrude_addition' or operation[0] == 'extrude_subtraction':
                    self.parse_extrude(Op, data[i-1])
                
                if operation[0] == 'fillet':
                    self.parse_fillet(Op)

        return

    def parse_sketch(self, Op):
        if 'radius' in Op['faces'][0]:
            self.parse_circle(Op)
            self.Op_idx += 1
            return 

        point_list = [vert['coordinates'] for vert in Op['vertices']]
        
        new_point_list = [point_list[0]]  # Start with the first point
        for i in range(1, len(point_list)):
            # Append each subsequent point twice
            new_point_list.append(point_list[i])
            new_point_list.append(point_list[i])
        
        # Add the first point again at the end to close the loop
        new_point_list.append(point_list[0])

        self.prev_sketch = proc_CAD.build123.protocol.build_sketch(self.Op_idx, new_point_list, self.output, self.data_directory)
        self.Op_idx += 1

    def parse_circle(self, Op):
        radius = Op['faces'][0]['radius']
        center = Op['faces'][0]['center']
        normal = Op['faces'][0]['normal']

        self.prev_sketch = proc_CAD.build123.protocol.build_circle(self.Op_idx, radius, center, normal, self.output, self.data_directory)
        self.Op_idx += 1
        
    def parse_extrude(self, Op, sketch_Op):

        isSubtract = (Op['operation'][0] == 'extrude_subtraction')
        sketch_point_list = [vert['coordinates'] for vert in sketch_Op['vertices']]
        sketch_face_normal = sketch_Op['faces'][0]['normal']
        extrude_amount = Op['operation'][2]

        expected_point = proc_CAD.helper.expected_extrude_point(sketch_point_list[0], sketch_face_normal, extrude_amount)
        
        if not isSubtract: 
            canvas_1 = proc_CAD.build123.protocol.test_extrude(self.prev_sketch, extrude_amount)
            canvas_2 = proc_CAD.build123.protocol.test_extrude(self.prev_sketch, -extrude_amount)

            if (canvas_1 is not None) and proc_CAD.helper.canvas_has_point(canvas_1, expected_point) :
                self.canvas = proc_CAD.build123.protocol.build_extrude(self.Op_idx, self.canvas, self.prev_sketch, extrude_amount, self.output, self.data_directory)
            if (canvas_2 is not None) and proc_CAD.helper.canvas_has_point(canvas_2, expected_point):
                self.canvas = proc_CAD.build123.protocol.build_extrude(self.Op_idx, self.canvas, self.prev_sketch, -extrude_amount, self.output, self.data_directory)
        else:
            self.canvas = proc_CAD.build123.protocol.build_subtract(self.Op_idx, self.canvas, self.prev_sketch, extrude_amount, self.output, self.data_directory)

        # proc_CAD.helper.print_canvas_points(self.canvas)

        self.Op_idx += 1
        
    def parse_fillet(self, Op):
        fillet_amount = Op['operation'][2]['amount']
        verts = Op['operation'][3]['old_verts_pos']

        target_edge = proc_CAD.helper.find_target_verts(verts, self.canvas.edges())

        if target_edge != None:
            self.canvas = proc_CAD.build123.protocol.build_fillet(self.Op_idx, self.canvas, target_edge, fillet_amount, self.output, self.data_directory)

    def is_valid_parse(self):
        return self.Op_idx == self.len_program


# Example usage:

def run(data_directory):
    file_path = os.path.join(os.path.dirname(__file__), 'canvas', 'Program.json')
    if data_directory:
        file_path = os.path.join(data_directory, 'Program.json')

    parsed_program_class = parsed_program(file_path, data_directory)
    parsed_program_class.read_json_file()
    
    return parsed_program_class.is_valid_parse()
