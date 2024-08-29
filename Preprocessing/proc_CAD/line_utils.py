
from itertools import combinations
from proc_CAD.basic_class import Face, Edge, Vertex

def midpoint_lines(edges):

    if len(edges) == 3:
        return triangle_midpoint_lines(edges)
    
    if len(edges) == 4:
        return rectangle_midpoint_lines(edges)

    return []


def triangle_midpoint_lines(edges):
    """
    Computes the midpoint of the third edge and creates a new edge that connects
    this midpoint to the vertex formed by the two edges of equal length in the triangle.
    """
    # Step 1: Extract idx1 from the ID of the first edge
    first_edge_id = edges[0].id
    idx1 = first_edge_id.split('_')[1]  # Extract the index after 'edge_'
    mp = "mp"  # Label for the midpoint

    # Step 2: Identify the two edges with the same length and the third edge
    edge_lengths = [(edge, ((edge.vertices[0].position[0] - edge.vertices[1].position[0])**2 +
                            (edge.vertices[0].position[1] - edge.vertices[1].position[1])**2 +
                            (edge.vertices[0].position[2] - edge.vertices[1].position[2])**2) ** 0.5) 
                    for edge in edges]

    # Sort edges based on their lengths
    edge_lengths.sort(key=lambda x: x[1])

    # Two edges with equal length
    edge1, edge2 = edge_lengths[0][0], edge_lengths[1][0]

    # Third edge
    third_edge = edge_lengths[2][0]

    # Step 3: Find the midpoint of the third edge
    point1 = third_edge.vertices[0].position
    point2 = third_edge.vertices[1].position
    midpoint_coords = tuple((point1[i] + point2[i]) / 2 for i in range(3))

    # Create a new vertex for the midpoint
    vertex_id = f'vert_{idx1}_{mp}_0'
    midpoint_vertex = Vertex(id=vertex_id, position=midpoint_coords)

    # Step 4: Determine the shared vertex of the two equal-length edges
    common_vertex = None
    for v1 in edge1.vertices:
        if v1 in edge2.vertices:
            common_vertex = v1
            break

    # Step 5: Create a new edge connecting the midpoint to the common vertex
    edge_id = f"edge_{idx1}_{mp}_0"
    new_edge = Edge(id=edge_id, vertices=(midpoint_vertex, common_vertex))

    # Return the new edge
    return [new_edge]



def rectangle_midpoint_lines(edges):
    """
    Computes the midpoints of the edges and creates new edge objects that connect these midpoints
    in a manner parallel to the rectangle's edges.
    """

    # Step 1: Extract idx1 from the ID of the first edge
    first_edge_id = edges[0].id
    idx1 = first_edge_id.split('_')[1]  # Extract the index after 'edge_'
    mp = "mp"  # Label for the midpoint

    # Step 2: Calculate midpoints of the edges
    midpoints = []
    current_vertex_idx = 0  # Start the new vertex index from 0

    for edge in edges:
        # Extract the 3D positions of the two vertices of the edge
        point1 = edge.vertices[0].position
        point2 = edge.vertices[1].position

        # Calculate the midpoint of the edge
        midpoint_coords = tuple((point1[i] + point2[i]) / 2 for i in range(3))
        
        # Create a new vertex for the midpoint
        vertex_id = f'vert_{idx1}_{mp}_{current_vertex_idx}'
        midpoint_vertex = Vertex(id=vertex_id, position=midpoint_coords)
        midpoints.append(midpoint_vertex)
        current_vertex_idx += 1  # Increment vertex index

    # Step 3: Find lines connecting midpoints that differ in exactly one coordinate
    midpoint_edges = []
    current_edge_idx = 0  # Start the new edge index from 0

    for mp1, mp2 in combinations(midpoints, 2):
        # Check if the two midpoints differ in exactly one coordinate
        differences = sum(1 for i in range(3) if mp1.position[i] != mp2.position[i])
        if differences == 1:  # Differ in exactly one coordinate
            # Create a new edge object
            edge_id = f"edge_{idx1}_{mp}_{current_edge_idx}"
            new_edge = Edge(id=edge_id, vertices=(mp1, mp2))
            midpoint_edges.append(new_edge)
            current_edge_idx += 1  # Increment the edge index for the next edge

    return midpoint_edges

def diagonal_lines(edges):

    """
    Creates diagonal lines by connecting the diagonal vertices of a rectangle formed by the given edges.
    """

    if len(edges) == 3:
        return []

    # Step 1: Extract idx1 from the ID of the first edge
    first_edge_id = edges[0].id
    idx1 = first_edge_id.split('_')[1]  # Extract the index after 'edge_'
    label = "diagonal"  # Label for diagonal lines

    # Step 2: Collect all unique vertices
    vertices = set()
    for edge in edges:
        vertices.update(edge.vertices)

    vertices = list(vertices)  # Convert the set back to a list for indexing

    # Step 3: Find diagonals
    diagonal_edges = []
    current_edge_idx = 0  # Start the new edge index from 0

    # Use the first vertex and find a diagonal vertex not on the same edge
    v1 = vertices[0]
    # Find a vertex not on the same edge as v1
    for v2 in vertices[1:]:
        if not any(v1 in edge.vertices and v2 in edge.vertices for edge in edges):
            # v1 and v2 are diagonal vertices
            edge_id = f"edge_{idx1}_{label}_{current_edge_idx}"
            new_edge = Edge(id=edge_id, vertices=(v1, v2))
            diagonal_edges.append(new_edge)
            current_edge_idx += 1
            break

    # The remaining two vertices form the other diagonal
    remaining_vertices = [v for v in vertices if v not in (v1, v2)]
    if len(remaining_vertices) == 2:
        v3, v4 = remaining_vertices
        edge_id = f"edge_{idx1}_{label}_{current_edge_idx}"
        new_edge = Edge(id=edge_id, vertices=(v3, v4))
        diagonal_edges.append(new_edge)

    return diagonal_edges
