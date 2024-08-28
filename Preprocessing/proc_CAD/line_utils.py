
from itertools import combinations
from proc_CAD.basic_class import Face, Edge, Vertex

def midpoint_lines(edges):
    """
    Computes the midpoints of the edges and creates new edge objects that connect these midpoints
    in a manner parallel to the rectangle's edges.

    :param edges: A list of edge objects. Each edge object should have a 'vertices' attribute,
                  which is a list or tuple containing two vertex objects. Each vertex object
                  should have an 'id' and a 'position' attribute representing its 3D coordinates (x, y, z).
    :return: A list of new edge objects created from the midpoints.
    """
    # Step 1: Extract idx1 from the ID of the first edge
    first_edge_id = edges[0].id
    idx1 = first_edge_id.split('_')[1]  # Extract the index after 'edge_'

    # Step 2: Calculate midpoints of the edges
    midpoints = []
    current_vertex_idx = len(edges)  # Start the new vertex index from the length of the existing edges

    for edge in edges:
        # Extract the 3D positions of the two vertices of the edge
        point1 = edge.vertices[0].position
        point2 = edge.vertices[1].position

        # Calculate the midpoint of the edge
        midpoint_coords = tuple((point1[i] + point2[i]) / 2 for i in range(3))
        
        # Create a new vertex for the midpoint
        vertex_data = {'id': f'vertex_{idx1}_{current_vertex_idx}', 'coordinates': midpoint_coords}
        midpoint_vertex = Vertex(id=vertex_data['id'], position=vertex_data['coordinates'])
        midpoints.append(midpoint_vertex)
        current_vertex_idx += 1  # Increment vertex index

    # Step 3: Find lines connecting midpoints that are parallel to the rectangle's edges
    midpoint_edges = []
    current_edge_idx = len(edges)  # Start the new edge index from the length of existing edges

    for mp1, mp2 in combinations(midpoints, 2):
        # Check if the two midpoints are aligned parallel to the rectangle's edges
        if mp1.position[0] == mp2.position[0] or mp1.position[1] == mp2.position[1] or mp1.position[2] == mp2.position[2]:
            # Create a new edge object
            edge_id = f"edge_{idx1}_{current_edge_idx}"
            new_edge = Edge(id=edge_id, vertices=(mp1, mp2))
            midpoint_edges.append(new_edge)
            current_edge_idx += 1  # Increment the edge index for the next edge

    return midpoint_edges
