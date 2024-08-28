
from itertools import combinations

def midpoint_lines(edges):

    # Step 1: Calculate midpoints of the edges
    midpoints = []
    for edge in edges:
        # Extract the 3D positions of the two vertices of the edge
        point1 = edge.vertices[0].position
        point2 = edge.vertices[1].position

        # Calculate the midpoint of the edge
        midpoint = tuple((point1[i] + point2[i]) / 2 for i in range(3))
        midpoints.append(midpoint)

    # Step 2: Find lines connecting midpoints that are parallel to the rectangle's edges
    midpoint_lines = []
    for mp1, mp2 in combinations(midpoints, 2):
        # Check if the two midpoints are aligned parallel to the rectangle's edges
        if mp1[0] == mp2[0] or mp1[1] == mp2[1] or mp1[2] == mp2[2]:  # Same x, y, or z
            midpoint_lines.append((mp1, mp2))

    return midpoint_lines
