import Models.operation_model
import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network

import Models.sketch_arguments.face_aggregate
import Models.sketch_arguments.sketch_model_2


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


SBGCN_model = Preprocessing.SBGCN.SBGCN_network.FaceEdgeVertexGCN()
stroke_embed_model = Models.sketch_arguments.sketch_model_2.StrokeEmbeddingNetwork()
plane_embed_model = Models.sketch_arguments.sketch_model_2.PlaneEmbeddingNetwork()
cross_attention_model = Models.sketch_arguments.sketch_model_2.FaceBrepAttention()

SBGCN_model.to(device)
stroke_embed_model.to(device)
plane_embed_model.to(device)
cross_attention_model.to(device)

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction_2')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    # Load models if they exist
    if os.path.exists(os.path.join(save_dir, 'SBGCN_model.pth')):
        SBGCN_model.load_state_dict(torch.load(os.path.join(save_dir, 'SBGCN_model.pth')))
        print("Loaded SBGCN_model")

    if os.path.exists(os.path.join(save_dir, 'stroke_embed_model.pth')):
        stroke_embed_model.load_state_dict(torch.load(os.path.join(save_dir, 'stroke_embed_model.pth')))
        print("Loaded stroke_embed_model")

    if os.path.exists(os.path.join(save_dir, 'plane_embed_model.pth')):
        plane_embed_model.load_state_dict(torch.load(os.path.join(save_dir, 'plane_embed_model.pth')))
        print("Loaded plane_embed_model")

    if os.path.exists(os.path.join(save_dir, 'cross_attention_model.pth')):    
        cross_attention_model.load_state_dict(torch.load(os.path.join(save_dir, 'cross_attention_model.pth')))
        print("Loaded cross_attention_model")


def save_models():
    torch.save(SBGCN_model.state_dict(), os.path.join(save_dir, 'SBGCN_model.pth'))
    torch.save(stroke_embed_model.state_dict(), os.path.join(save_dir, 'stroke_embed_model.pth'))
    torch.save(plane_embed_model.state_dict(), os.path.join(save_dir, 'plane_embed_model.pth'))
    torch.save(cross_attention_model.state_dict(), os.path.join(save_dir, 'cross_attention_model.pth'))

    print("Saved models.")


def vis_gt(node_features, face_to_stroke, chosen_face):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove the first dimension
    node_features = node_features.squeeze(0)

    # Plot all strokes in blue
    for stroke in node_features:
        start = stroke[:3].numpy()
        end = stroke[3:].numpy()
        
        # Plot the line segment for the stroke in blue
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], marker='o', color='blue')

    # Find the chosen face
    chosen_face_index = torch.where(chosen_face == 1)[0]
    
    if len(chosen_face_index) > 0:
        chosen_face_index = chosen_face_index[0].item()  # Get the first chosen face index

        # Find the strokes for the chosen face
        chosen_strokes = face_to_stroke[chosen_face_index]

        # Plot the chosen strokes in red
        for stroke_index in chosen_strokes:
            stroke = node_features[stroke_index]
            start = stroke[0][:3].numpy()
            end = stroke[0][3:].numpy()
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], marker='o', color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    

def vis_output(node_features, face_to_stroke, chosen_face):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove the first dimension
    node_features = node_features.squeeze(0)

    # Plot all strokes in blue
    for stroke in node_features:
        start = stroke[:3].numpy()
        end = stroke[3:].numpy()
        
        # Plot the line segment for the stroke in blue
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], marker='o', color='blue')

    # Find the chosen face
    chosen_face_index = torch.where(chosen_face > 0.5)[0]
    
    if len(chosen_face_index) > 0:
        chosen_face_index = chosen_face_index[0].item()  # Get the first chosen face index

        # Find the strokes for the chosen face
        chosen_strokes = face_to_stroke[chosen_face_index]

        # Plot the chosen strokes in red
        for stroke_index in chosen_strokes:
            stroke = node_features[stroke_index]
            start = stroke[0][:3].numpy()
            end = stroke[0][3:].numpy()
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], marker='o', color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def train():
    # Define training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(SBGCN_model.parameters()) +
        list(stroke_embed_model.parameters()) +
        list(plane_embed_model.parameters()) +
        list(cross_attention_model.parameters()),
        lr=0.001
    )

    epochs = 20

    # Create a DataLoader
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/train_dataset')

    # Filter to only keep
    good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 1]
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
        SBGCN_model.train()
        stroke_embed_model.train()
        plane_embed_model.train()
        cross_attention_model.train()

        # Training loop
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, _, _, operations_order_matrix, face_to_stroke, program, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

            # Move data to device
            node_features = node_features.to(torch.float32).to(device)
            face_to_stroke = [[indices.to(device) for indices in face] for face in face_to_stroke]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # 1) Embed the strokes
            stroke_embed = stroke_embed_model(node_features)

            # 2) Find all possible faces
            # This is given by face_to_stroke

            # 3) For each face, build the embedding
            face_embed = plane_embed_model(face_to_stroke, stroke_embed)

            # 4) Prepare brep_embedding
            if face_features.shape[1] == 0:
                # is empty program
                brep_embedding = torch.zeros(1, 1, 32, device=device)
            else:
                brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
                            edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
                
                brep_graph.to_device(device)
                face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)
                brep_embedding = torch.cat((face_embedding, edge_embedding, vertex_embedding), dim=1)

            # 5) Do cross attention on face_embedding and brep_embedding
            output = cross_attention_model(face_embed, brep_embedding)

            # 6) Prepare ground_truth
            target_op_index = len(program[0]) - 1
            op_to_index_matrix = operations_order_matrix
            kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
            gt_matrix = Models.sketch_arguments.face_aggregate.build_gt_matrix(kth_operation, face_to_stroke)

            # 7) Compute the loss
            loss = criterion(output, gt_matrix)

            # 8) Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Compute average training loss
        train_loss /= len(train_loader)



        # Validation loop
        SBGCN_model.eval()
        stroke_embed_model.eval()
        plane_embed_model.eval()
        cross_attention_model.eval()

        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                node_features, _, _, operations_order_matrix, face_to_stroke, program, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

                # Move data to device
                node_features = node_features.to(torch.float32).to(device)
                face_to_stroke = [[indices.to(device) for indices in face] for face in face_to_stroke]

                # 1) Embed the strokes
                stroke_embed = stroke_embed_model(node_features)

                # 2) Find all possible faces
                # This is given by face_to_stroke

                # 3) For each face, build the embedding
                face_embed = plane_embed_model(face_to_stroke, stroke_embed)

                # 4) Prepare brep_embedding
                if face_features.shape[1] == 0:
                    # is empty program
                    brep_embedding = torch.zeros(1, 1, 32, device=device)
                else:
                    brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
                                edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
                    
                    brep_graph.to_device(device)
                    face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)
                    brep_embedding = torch.cat((face_embedding, edge_embedding, vertex_embedding), dim=1)

                # 5) Do cross attention on face_embedding and brep_embedding
                output = cross_attention_model(face_embed, brep_embedding)

                # 6) Prepare ground_truth
                target_op_index = len(program[0]) - 1
                op_to_index_matrix = operations_order_matrix
                kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
                gt_matrix = Models.sketch_arguments.face_aggregate.build_gt_matrix(kth_operation, face_to_stroke)

                # 7) Compute the loss
                loss = criterion(output, gt_matrix)

                val_loss += loss.item()

        # Compute average validation loss
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss  
            save_models()



def eval():
    # Load models
    load_models()

    # Create a DataLoader for the evaluation dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/eval_dataset')

    # Filter to only keep good data
    good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 1]
    filtered_dataset = Subset(dataset, good_data_indices)
    print(f"Total number of sketch data: {len(filtered_dataset)}")

    eval_loader = DataLoader(filtered_dataset, batch_size=1, shuffle=False)

    # Set models to evaluation mode
    SBGCN_model.eval()
    stroke_embed_model.eval()
    plane_embed_model.eval()
    cross_attention_model.eval()

    eval_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            node_features, _, _, operations_order_matrix, face_to_stroke, program, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

            # Move data to device
            node_features = node_features.to(torch.float32).to(device)
            face_to_stroke = [[indices.to(device) for indices in face] for face in face_to_stroke]

            # 1) Embed the strokes
            stroke_embed = stroke_embed_model(node_features)

            # 2) Find all possible faces
            # This is given by face_to_stroke

            # 3) For each face, build the embedding
            face_embed = plane_embed_model(face_to_stroke, stroke_embed)

            # 4) Prepare brep_embedding
            if face_features.shape[1] == 0:
                # is empty program
                brep_embedding = torch.zeros(1, 1, 32, device=device)
            else:
                brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
                            edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
                
                brep_graph.to_device(device)
                face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)
                brep_embedding = torch.cat((face_embedding, edge_embedding, vertex_embedding), dim=1)

            # 5) Do cross attention on face_embedding and brep_embedding
            output = cross_attention_model(face_embed, brep_embedding)

            # 6) Prepare ground_truth
            target_op_index = len(program[0]) - 1
            op_to_index_matrix = operations_order_matrix
            kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
            gt_matrix = Models.sketch_arguments.face_aggregate.build_gt_matrix(kth_operation, face_to_stroke)

            # 7) Compute the loss
            loss = criterion(output, gt_matrix)

            eval_loss += loss.item()

            print("output", output)
            vis_gt(node_features, face_to_stroke, gt_matrix)
            vis_output(node_features, face_to_stroke, output)
            break

    # Compute average evaluation loss
    eval_loss /= len(eval_loader)
    print(f"Evaluation loss: {eval_loss:.4f}")




#---------------------------------- Public Functions ----------------------------------#

eval()
