import numpy as np
import proc_CAD.generate_program
import random



def random_program(steps = 3):
    canvas_class = proc_CAD.generate_program.Brep()

    canvas_class.init_sketch_op()
    canvas_class.add_extrude_add_op()
    
    for _ in range(steps - 1):
        canvas_class.regular_sketch_op()
        canvas_class.add_extrude_add_op()

        fillet_times = random.randint(0, 2)
        for _ in range(fillet_times):
            canvas_class.random_fillet()

    canvas_class.write_to_json()

def simple_gen():
    canvas_class = proc_CAD.generate_program.Brep()
    canvas_class.init_sketch_op()
    canvas_class.add_extrude_add_op()
    canvas_class.random_fillet()

    canvas_class.write_to_json()
