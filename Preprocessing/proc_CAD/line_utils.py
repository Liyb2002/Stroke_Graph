
from itertools import combinations
from proc_CAD.basic_class import Face, Edge, Vertex


# -------------------- Midpoint Lines -------------------- #

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



# -------------------- Diagonal Lines -------------------- #
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



# -------------------- Projection Lines -------------------- #
def projection_lines(edges):
    """
    Computes the midpoints of the extruding lines (edges) and connects these midpoints 
    to form a sketch that is parallel to the original sketch.
    """
    # Step 1: Extract idx1 from the ID of the first edge
    first_edge_id = edges[0].id
    idx1 = first_edge_id.split('_')[1]  # Extract the index after 'edge_'
    mp = "projection"

    # Determine the number of edges to consider (first half)
    half_length = len(edges) // 2

    # Step 2: Calculate midpoints of the first half of the extruding lines
    midpoints = []
    current_vertex_idx = 0  # Start the new vertex index from 0

    for i in range(half_length, len(edges)):
        edge = edges[i]
        
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

    # Step 3: Connect the midpoints to form the new sketch
    midpoint_edges = []
    current_edge_idx = 0  # Start the new edge index from 0

    # Assuming the midpoints should be connected in sequence
    for i in range(len(midpoints)):
        mp1 = midpoints[i]
        mp2 = midpoints[(i + 1) % len(midpoints)]  # Next midpoint, wrapping around

        # Create a new edge object connecting adjacent midpoints
        edge_id = f"edge_{idx1}_{mp}_{current_edge_idx}"
        new_edge = Edge(id=edge_id, vertices=(mp1, mp2))
        midpoint_edges.append(new_edge)
        current_edge_idx += 1  # Increment the edge index for the next edge

    return midpoint_edges



# -------------------- bounding_box Lines -------------------- #

def bounding_box_lines(edges):
    """
    Creates a bounding box around the given edges. The bounding box is defined by 12 lines 
    connecting the minimum and maximum x, y, and z coordinates. If a line already exists 
    in the edges, it will not be recreated.
    """

    if len(edges) != 6:
        return []

    # Initialize min and max values
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    # Step 1: Find the min and max x, y, z values
    for edge in edges:
        for vertex in edge.vertices:
            pos = vertex.position
            min_x, max_x = min(min_x, pos[0]), max(max_x, pos[0])
            min_y, max_y = min(min_y, pos[1]), max(max_y, pos[1])
            min_z, max_z = min(min_z, pos[2]), max(max_z, pos[2])

    # Define the 8 vertices of the bounding box
    bbox_vertices = [
        (min_x, min_y, min_z), (min_x, min_y, max_z),
        (min_x, max_y, min_z), (min_x, max_y, max_z),
        (max_x, min_y, min_z), (max_x, min_y, max_z),
        (max_x, max_y, min_z), (max_x, max_y, max_z)
    ]

    # Generate all 12 edges of the bounding box
    bbox_edges_indices = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Convert the edges to a set of tuples for quick comparison
    existing_edges = set()
    for edge in edges:
        vertex_positions = tuple(sorted((tuple(edge.vertices[0].position), tuple(edge.vertices[1].position))))
        existing_edges.add(vertex_positions)

    # Step 2: Create the edges of the bounding box
    bounding_box_edges = []
    current_edge_idx = 0  # Index for new edges
    for v1_idx, v2_idx in bbox_edges_indices:
        v1, v2 = bbox_vertices[v1_idx], bbox_vertices[v2_idx]
        vertex_positions = tuple(sorted((v1, v2)))

        # Check if the edge already exists
        if vertex_positions not in existing_edges:
            # Create two new vertices
            vertex1 = Vertex(id=f'vert_bbox_{current_edge_idx}_0', position=v1)
            vertex2 = Vertex(id=f'vert_bbox_{current_edge_idx}_1', position=v2)

            # Create a new edge connecting these vertices
            edge_id = f'edge_bbox_{current_edge_idx}'
            new_edge = Edge(id=edge_id, vertices=(vertex1, vertex2))
            bounding_box_edges.append(new_edge)
            current_edge_idx += 1

    return bounding_box_edges


# -------------------- whole_bounding_box Lines -------------------- #

def whole_bounding_box_lines(all_edges):
    """
    Creates a bounding box around all the edges provided in the set `all_edges`. 
    The bounding box is defined by 12 lines connecting the minimum and maximum x, y, and z coordinates. 
    If a line is already inside `all_edges`, it will not be recreated.
    """
    # Initialize min and max values
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    # Step 1: Find the min and max x, y, z values from all edges
    for other_edge_id, other_edge in all_edges.items():
        for vertex in other_edge.vertices:
            pos = vertex.position
            min_x, max_x = min(min_x, pos[0]), max(max_x, pos[0])
            min_y, max_y = min(min_y, pos[1]), max(max_y, pos[1])
            min_z, max_z = min(min_z, pos[2]), max(max_z, pos[2])

    # Define the 8 vertices of the bounding box
    bbox_vertices = [
        (min_x, min_y, min_z), (min_x, min_y, max_z),
        (min_x, max_y, min_z), (min_x, max_y, max_z),
        (max_x, min_y, min_z), (max_x, min_y, max_z),
        (max_x, max_y, min_z), (max_x, max_y, max_z)
    ]

    # Generate all 12 edges of the bounding box
    bbox_edges_indices = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Convert the edges to a set of tuples for quick comparison
    existing_edges = set()
    for other_edge_id, other_edge in all_edges.items():
        vertex_positions = tuple(sorted((tuple(other_edge.vertices[0].position), tuple(other_edge.vertices[1].position))))
        existing_edges.add(vertex_positions)

    # Step 2: Create the edges of the bounding box
    bounding_box_edges = []
    current_edge_idx = 0  # Index for new edges
    for v1_idx, v2_idx in bbox_edges_indices:
        v1, v2 = bbox_vertices[v1_idx], bbox_vertices[v2_idx]
        vertex_positions = tuple(sorted((v1, v2)))

        # Check if the edge already exists
        if vertex_positions not in existing_edges:
            # Create two new vertices
            vertex1 = Vertex(id=f'whole_vert_bbox_{current_edge_idx}_0', position=v1)
            vertex2 = Vertex(id=f'whole_vert_bbox_{current_edge_idx}_1', position=v2)

            # Create a new edge connecting these vertices
            edge_id = f'whole_edge_bbox_{current_edge_idx}'
            new_edge = Edge(id=edge_id, vertices=(vertex1, vertex2))
            bounding_box_edges.append(new_edge)
            current_edge_idx += 1

    return bounding_box_edges

