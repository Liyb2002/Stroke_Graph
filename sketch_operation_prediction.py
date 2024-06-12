import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network

import Encoders.gnn.gnn
import Encoders.program_encoder.program_encoder

import Models.sketch_model


from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from itertools import combinations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the neural networks
graph_embedding_model = Encoders.gnn.gnn.SemanticModule()
SBGCN_model = Preprocessing.SBGCN.SBGCN_network.FaceEdgeVertexGCN()
sketch_attention_model = Models.sketch_model.GraphBrepAttention()

graph_embedding_model.to(device)
SBGCN_model.to(device)
sketch_attention_model.to(device)

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
os.makedirs(save_dir, exist_ok=True)


def load_models():
    # Load models if they exist
    if os.path.exists(os.path.join(save_dir, 'graph_embedding_model.pth')):
        graph_embedding_model.load_state_dict(torch.load(os.path.join(save_dir, 'graph_embedding_model.pth')))
        print("Loaded graph_embedding_model")

    if os.path.exists(os.path.join(save_dir, 'SBGCN_model.pth')):
        SBGCN_model.load_state_dict(torch.load(os.path.join(save_dir, 'SBGCN_model.pth')))
        print("Loaded SBGCN_model")

    if os.path.exists(os.path.join(save_dir, 'sketch_attention_model.pth')):    
        sketch_attention_model.load_state_dict(torch.load(os.path.join(save_dir, 'sketch_attention_model.pth')))
        print("Loaded sketch_attention_model")


def save_models():
    torch.save(graph_embedding_model.state_dict(), os.path.join(save_dir, 'graph_embedding_model.pth'))
    torch.save(SBGCN_model.state_dict(), os.path.join(save_dir, 'SBGCN_model.pth'))
    torch.save(sketch_attention_model.state_dict(), os.path.join(save_dir, 'sketch_attention_model.pth'))
    print("Saved models.")


def get_kth_operation(op_to_index_matrix, k):    
    squeezed_matrix = op_to_index_matrix.squeeze(0)
    kth_operation = squeezed_matrix[:, k].unsqueeze(1)

    return kth_operation


def face_aggregate(strokes, stroke_features):
    # Ensure strokes and stroke_features are tensors
    strokes = strokes.clone().detach()
    stroke_features = stroke_features.clone().detach().squeeze(0)
    
    # Get the coordinates of the chosen strokes
    chosen_indices = (strokes > 0.5).nonzero(as_tuple=True)[0]
    chosen_strokes = stroke_features[chosen_indices]

    def find_coplanar_lines(lines):
        subsets = []
        lines = lines.numpy()

        for i, line in enumerate(lines):
            start, end = line[:3], line[3:]
            coplanar_set = [line]

            for j, other_line in enumerate(lines):
                if i != j:
                    other_start, other_end = other_line[:3], other_line[3:]
                    # Check if the lines are coplanar by having two common coordinates
                    if (start[0] == other_start[0] and end[0] == other_end[0]) or \
                       (start[1] == other_start[1] and end[1] == other_end[1]) or \
                       (start[2] == other_start[2] and end[2] == other_end[2]):
                        coplanar_set.append(other_line)

            if len(coplanar_set) > 2:
                subsets.append(coplanar_set)

        return subsets

    def is_connected(subset):
        graph = {}
        for line in subset:
            start = tuple(line[:3])
            end = tuple(line[3:])
            if start not in graph:
                graph[start] = []
            if end not in graph:
                graph[end] = []
            graph[start].append(end)
            graph[end].append(start)

        visited = set()
        stack = [tuple(subset[0][:3])]

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(graph[node])

        return len(visited) == len(graph)

    def find_connected_subsets(subsets):
        connected_subsets = []
        for subset in subsets:
            connected_points = set()
            new_subset = []
            for line in subset:
                start, end = tuple(line[:3]), tuple(line[3:])
                if start in connected_points or end in connected_points or not connected_points:
                    new_subset.append(start)
                    new_subset.append(end)
                    connected_points.update([start, end])
                else:
                    if len(new_subset) > 2 and is_connected(new_subset):
                        connected_subsets.append(list(set(new_subset)))
                    new_subset = [start, end]
                    connected_points = {start, end}
            if len(new_subset) > 2 and is_connected(new_subset):
                connected_subsets.append(list(set(new_subset)))
        return connected_subsets

    def remove_contained_subsets(subsets):
    # Convert each subset to a set for easier comparison
        subset_sets = [set(subset) for subset in subsets]
        
        # Create a list to store subsets that are not fully contained in others
        result = []
        
        for i, subset in enumerate(subset_sets):
            is_contained = False
            for j, other_subset in enumerate(subset_sets):
                if i != j and subset.issubset(other_subset):
                    is_contained = True
                    break
            if not is_contained:
                result.append(subsets[i])
    
        return result

    def reorder_points(points):
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        ordered_points = [points[0]]
        points = np.delete(points, 0, axis=0)
        
        while points.size > 0:
            last_point = ordered_points[-1]
            distances = np.linalg.norm(points - last_point, axis=1)
            nearest_idx = np.argmin(distances)
            ordered_points.append(points[nearest_idx])
            points = np.delete(points, nearest_idx, axis=0)
        
        return ordered_points

    coplanar_subsets = find_coplanar_lines(chosen_strokes)
    connected_coplanar_subsets = find_connected_subsets(coplanar_subsets)

    # Remove duplicate point sets from connected_coplanar_subsets
    unique_connected_coplanar_subsets = []
    for subset in connected_coplanar_subsets:
        if subset not in unique_connected_coplanar_subsets and len(subset) > 2:
            unique_connected_coplanar_subsets.append(subset)

    unique_connected_subsets = remove_contained_subsets(unique_connected_coplanar_subsets)
    ordered_faces = []
    for subset in unique_connected_subsets:
        ordered_faces.append(reorder_points(subset))

    return ordered_faces



def vis_gt(strokes, stroke_features):
    # Ensure strokes and stroke_features are tensors
    strokes = strokes.clone().detach()
    stroke_features = stroke_features.clone().detach().squeeze(0)
    
    num_chosen_strokes = torch.sum(strokes).item()
    print(f"Number of chosen strokes in the strokes matrix: {num_chosen_strokes}")

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    num_strokes = stroke_features.shape[0]
    
    for i in range(num_strokes):
        x = [stroke_features[i, 0].item(), stroke_features[i, 3].item()]
        y = [stroke_features[i, 1].item(), stroke_features[i, 4].item()]
        z = [stroke_features[i, 2].item(), stroke_features[i, 5].item()]
        
        color = 'red' if strokes[i, 0].item() == 1 else 'blue'
        ax.plot(x, y, z, color=color)
    
    plt.show()


def vis_predict(strokes, stroke_features):
    # Ensure strokes and stroke_features are tensors
    strokes = strokes.clone().detach()
    stroke_features = stroke_features.clone().detach().squeeze(0)
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    num_strokes = stroke_features.shape[0]
    
    for i in range(num_strokes):
        x = [stroke_features[i, 0].item(), stroke_features[i, 3].item()]
        y = [stroke_features[i, 1].item(), stroke_features[i, 4].item()]
        z = [stroke_features[i, 2].item(), stroke_features[i, 5].item()]
        
        color = 'red' if strokes[i, 0].item() > 0.5 else 'blue'
        # ax.plot(x, y, z, color=color)
    
    chosen_strokes_sets = face_aggregate(strokes, stroke_features)
    num_faces = len(chosen_strokes_sets)

    for item in chosen_strokes_sets:
        print("item", item)

    print(f"Number of faces (sets of coplanar strokes): {num_faces}")

    for chosen_points in chosen_strokes_sets:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Separate the coordinates
        x = [point[0] for point in chosen_points]
        y = [point[1] for point in chosen_points]
        z = [point[2] for point in chosen_points]
        
        # Connect the last point with the first point
        x.append(x[0])
        y.append(y[0])
        z.append(z[0])
        
        # Plot the lines connecting the points
        ax.plot(x, y, z, marker='o', color='green')
        
        # Optionally, scatter the points for better visibility
        ax.scatter(x, y, z, color='red')
        
        plt.show()


def train():

    # Define training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(graph_embedding_model.parameters()) + 
        list(SBGCN_model.parameters()),
        lr=0.001
    )

    epochs = 20

    # Create a DataLoader
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/train_dataset')

    # Filter to only keep
    good_data_indices = [i for i, data in enumerate(dataset) if data[4][-1] == 1]
    filtered_dataset = Subset(dataset, good_data_indices)
    print(f"Total number of sketch data: {len(filtered_dataset)}")

    # Split dataset into training and validation
    train_size = int(0.8 * len(filtered_dataset))
    val_size = len(filtered_dataset) - train_size
    train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        graph_embedding_model.train()
        SBGCN_model.train()
        
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, program, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch
            
            # to device 
            node_features = node_features.to(torch.float32).to(device)
            operations_matrix = operations_matrix.to(torch.float32).to(device)
            intersection_matrix = intersection_matrix.to(torch.float32).to(device)
            operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)

            gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.to_device(device)
            graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)

            if face_features.shape[1] == 0:
                # is empty program
                brep_embedding = torch.zeros(1, 1, 32, device=device)
            else:
                brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
                            edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
                
                brep_graph.to_device(device)
                face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)
                brep_embedding = torch.cat((face_embedding, edge_embedding, vertex_embedding), dim=1)

            output = sketch_attention_model(graph_embedding, brep_embedding)

            # prepare ground_truth
            target_op_index = len(program[0])-1
            op_to_index_matrix = gnn_graph['stroke'].z
            gt_matrix = get_kth_operation(op_to_index_matrix, target_op_index)

            output = output.view(-1, 1).to(torch.float32)
            gt_matrix = gt_matrix.view(-1, 1).to(torch.float32)

            loss = criterion(output, gt_matrix)
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}")

        graph_embedding_model.eval()
        SBGCN_model.eval()
        
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                node_features, operations_matrix, intersection_matrix, operations_order_matrix, program, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch
                
                # to device 
                node_features = node_features.to(torch.float32).to(device)
                operations_matrix = operations_matrix.to(torch.float32).to(device)
                intersection_matrix = intersection_matrix.to(torch.float32).to(device)
                operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)

                gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
                gnn_graph.to_device(device)
                graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)

                if face_features.shape[1] == 0:
                    # is empty program
                    brep_embedding = torch.zeros(1, 1, 32, device=device)
                else:
                    brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
                                edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
                    
                    brep_graph.to_device(device)
                    face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)
                    brep_embedding = torch.cat((face_embedding, edge_embedding, vertex_embedding), dim=1)

                output = sketch_attention_model(graph_embedding, brep_embedding)

                # prepare ground_truth
                target_op_index = len(program[0])-1
                op_to_index_matrix = gnn_graph['stroke'].z
                gt_matrix = get_kth_operation(op_to_index_matrix, target_op_index).to(device)

                output = output.view(-1, 1).to(torch.float32)
                gt_matrix = gt_matrix.view(-1, 1).to(torch.float32)

                loss = criterion(output, gt_matrix)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}")

        # Checkpoint saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_models()


def eval(vis=True):
    load_models()

    graph_embedding_model.eval()
    SBGCN_model.eval()
    sketch_attention_model.eval()

    # Define evaluation criterion
    criterion = nn.BCEWithLogitsLoss()

    # Create DataLoader
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/eval_dataset')

    # Filter to only keep good data
    good_data_indices = [i for i, data in enumerate(dataset) if data[4][-1] == 1]
    filtered_dataset = Subset(dataset, good_data_indices)
    print(f"Total number of sketch data: {len(filtered_dataset)}")

    eval_loader = DataLoader(filtered_dataset, batch_size=1, shuffle=False)

    total_eval_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, program, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

            # to device 
            node_features = node_features.to(torch.float32).to(device)
            operations_matrix = operations_matrix.to(torch.float32).to(device)
            intersection_matrix = intersection_matrix.to(torch.float32).to(device)
            operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)

            gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.to_device(device)
            graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)

            if face_features.shape[1] == 0:
                # is empty program
                brep_embedding = torch.zeros(1, 1, 32, device=device)
            else:
                brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
                            edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
                
                brep_graph.to_device(device)
                face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)
                brep_embedding = torch.cat((face_embedding, edge_embedding, vertex_embedding), dim=1)

            output = sketch_attention_model(graph_embedding, brep_embedding)

            # prepare ground_truth
            target_op_index = len(program[0])-1
            op_to_index_matrix = gnn_graph['stroke'].z

            gt_matrix = get_kth_operation(op_to_index_matrix, target_op_index).to(device)

            output = output.view(-1, 1).to(torch.float32)
            gt_matrix = gt_matrix.view(-1, 1).to(torch.float32)

            loss = criterion(output, gt_matrix)
            total_eval_loss += loss.item()

            if vis:
                # vis_gt(gt_matrix, gnn_graph['stroke'].x)
                vis_predict(output, gnn_graph['stroke'].x)

                break

    avg_eval_loss = total_eval_loss / len(eval_loader)
    print(f"Evaluation Loss: {avg_eval_loss}")





#---------------------------------- Public Functions ----------------------------------#

eval()
