import random_gen
import numpy as np
import brep_class

canvas_class = brep_class.Brep()

# Example usage:

canvas_class.init_sketch_op()
canvas_class.add_extrude_add_op()
canvas_class.random_fillet()

# canvas_class.regular_sketch_op()
# canvas_class.add_extrude_add_op()
# canvas_class.random_fillet()

canvas_class.write_to_json()

