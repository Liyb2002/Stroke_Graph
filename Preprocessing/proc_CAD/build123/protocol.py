from pathlib import Path
from build123d import *
import os
import numpy as np

home_dir = Path(__file__).parent.parent


def build_sketch(count, Points_list, output):
    brep_dir = os.path.join(home_dir, "canvas", f"brep_{count}.stp")
    stl_dir = os.path.join(home_dir, "canvas", f"vis_{count}.stl")

    with BuildSketch():
        with BuildLine():
            lines = []
            for i in range(0, len(Points_list), 2):
                start_point_sublist = Points_list[i]
                end_point_sublist = Points_list[i+1]
                start_point = (start_point_sublist[0],
                               start_point_sublist[1], 
                               start_point_sublist[2])
                
                
                end_point = (end_point_sublist[0],
                            end_point_sublist[1], 
                            end_point_sublist[2])


                line = Line(start_point, end_point)
                lines.append(line)

        perimeter = make_face()

    if output:
        perimeter.export_stl(stl_dir)
        perimeter.export_brep(brep_dir)

    return perimeter

def build_circle(count, radius, point, normal, output):
    brep_dir = os.path.join(home_dir, "canvas", f"brep_{count}.stp")
    stl_dir = os.path.join(home_dir, "canvas", f"vis_{count}.stl")

    
    with BuildSketch(Plane(origin=(point[0], point[1], point[2]), z_dir=(normal[0], normal[1], normal[2])) )as perimeter:
        Circle(radius = 0.2)

    if output:
        perimeter.sketch.export_stl(stl_dir)
        perimeter.sketch.export_brep(brep_dir)

    return perimeter.sketch


def build_extrude(count, canvas, target_face, extrude_amount, output):
    stl_dir = os.path.join(home_dir, "canvas", f"vis_{count}.stl")
    step_dir = os.path.join(home_dir, "canvas", f"step_{count}.stp")

    if canvas != None:
        if extrude_amount <0:
            with canvas: 
                extrude_amount = -extrude_amount
                extrude( target_face, amount=extrude_amount, mode=Mode.SUBTRACT)
        else: 
            with canvas: 
                extrude( target_face, amount=extrude_amount)

    else:
        if extrude_amount <0:
            extrude_amount = -extrude_amount
        with BuildPart() as canvas:
            extrude( target_face, amount=extrude_amount)

    if output:
        canvas.part.export_stl(stl_dir)
        canvas.part.export_step(step_dir)

    return canvas


def build_fillet(count, canvas, target_edge, radius, output):
    print("build_fillet")
    stl_dir = os.path.join(home_dir, "canvas", f"vis_{count}.stl")
    step_dir = os.path.join(home_dir, "canvas", f"step_{count}.stp")

    with canvas:
        fillet(target_edge, radius)
    
    if output:
        canvas.part.export_stl(stl_dir)
        canvas.part.export_step(step_dir)

    return canvas