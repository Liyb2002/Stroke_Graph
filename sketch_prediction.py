import Preprocessing.dataloader
import Preprocessing.gnn_graph_full
import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network

import Models.sketch_model_helper
import Encoders.gnn_full.gnn
import Models.sketch_arguments.face_aggregate

import full_graph_train

from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


brep_graph_encoder = Encoders.gnn_full.gnn.SemanticModule()
brep_graph_decoder = Encoders.gnn_full.gnn.Sketch_brep_prediction()
strokes_decoder = Encoders.gnn_full.gnn.Final_stroke_finding()

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'stroke_choosing')
os.makedirs(save_dir, exist_ok=True)

def load_brep_models():
    # Load models if they exist
    brep_model_dir = os.path.join(current_dir, 'checkpoints', 'full_graph_sketch')

    if os.path.exists(os.path.join(brep_model_dir, 'graph_encoder.pth')):
        brep_graph_encoder.load_state_dict(torch.load(os.path.join(brep_model_dir, 'graph_encoder.pth')))
        print("Loaded graph_encoder")

    if os.path.exists(os.path.join(brep_model_dir, 'graph_decoder.pth')):
        brep_graph_decoder.load_state_dict(torch.load(os.path.join(brep_model_dir, 'graph_decoder.pth')))
        print("Loaded graph_decoder")


def load_models():
    # Load models if they exist
    if os.path.exists(os.path.join(save_dir, 'strokes_decoder.pth')):
        strokes_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'strokes_decoder.pth')))
        print("Loaded strokes_decoder")

def save_models():
    torch.save(strokes_decoder.state_dict(), os.path.join(save_dir, 'strokes_decoder.pth'))
    print("Saved models.")



# Define optimizer and loss function
optimizer = optim.Adam( strokes_decoder.parameters(), lr=0.0004)
loss_function = nn.BCELoss()

# Load the dataset
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/full_train_dataset')
good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 1]
filtered_dataset = Subset(dataset, good_data_indices)
print(f"Total number of sketch data: {len(filtered_dataset)}")

# Split the dataset into training and validation sets
train_size = int(0.8 * len(filtered_dataset))
val_size = len(filtered_dataset) - train_size
train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)




def train():
    load_brep_models()
    # Training and validation loop
    best_val_loss = float('inf')
    epochs = 30

    for epoch in range(epochs):
        # Training loop
        strokes_decoder.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, face_boundary_points, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

            if edge_features.shape[1] == 0:
                continue
            
            # get the predictions 
            brep_edges_weights = full_graph_train.predict_brep_edges(brep_graph_encoder, brep_graph_decoder, batch)

            # move tp device
            node_features = node_features.to(torch.float32).to(device).squeeze(0)
            operations_matrix = operations_matrix.to(torch.float32).to(device)
            intersection_matrix = intersection_matrix.to(torch.float32).to(device)
            operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)
            edge_features = edge_features.to(torch.float32).to(device).squeeze(0)


            gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            brep_stroke_connection_matrix, stroke_coplanar_matrix = gnn_graph.set_brep_connection(edge_features, face_feature_gnn_list)

            stroke_weights = Models.sketch_model_helper.integrate_brep_probs(brep_edges_weights, brep_stroke_connection_matrix, stroke_coplanar_matrix)

            x_dict = brep_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = strokes_decoder(x_dict, gnn_graph.edge_index_dict, stroke_weights)
            
            # Prepare gt
            target_op_index = len(program[0]) - 1
            op_to_index_matrix = operations_order_matrix
            kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
            
            loss = loss_function(output, kth_operation)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()


        strokes_decoder.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, face_boundary_points, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

                if edge_features.shape[1] == 0:
                    continue
                
                # get the predictions 
                brep_edges_weights = full_graph_train.predict_brep_edges(brep_graph_encoder, brep_graph_decoder, batch)

                # move tp device
                node_features = node_features.to(torch.float32).to(device).squeeze(0)
                operations_matrix = operations_matrix.to(torch.float32).to(device)
                intersection_matrix = intersection_matrix.to(torch.float32).to(device)
                operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)
                edge_features = edge_features.to(torch.float32).to(device).squeeze(0)


                gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
                brep_stroke_connection_matrix, stroke_coplanar_matrix = gnn_graph.set_brep_connection(edge_features, face_feature_gnn_list)

                stroke_weights = Models.sketch_model_helper.integrate_brep_probs(brep_edges_weights, brep_stroke_connection_matrix, stroke_coplanar_matrix)

                x_dict = brep_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
                output = strokes_decoder(x_dict, gnn_graph.edge_index_dict, stroke_weights)
                
                # Prepare gt
                target_op_index = len(program[0]) - 1
                op_to_index_matrix = operations_order_matrix
                kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
                
                loss = loss_function(output, kth_operation)
                total_val_loss += loss.item()

        if best_val_loss > total_val_loss:
            best_val_loss =  total_val_loss
            save_models()
        
        # Print epoch losses
        print(f"Epoch {epoch+1}: Train Loss = {total_train_loss/len(train_loader)}, Val Loss = {total_val_loss/len(val_loader)}")


def eval():
    load_brep_models()
    load_models()

    correct_predictions = 0
    total_predictions = 0
    for batch in tqdm(train_loader):
        node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, face_boundary_points, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

        if edge_features.shape[1] == 0:
            continue
        
        # get the predictions 
        brep_edges_weights = full_graph_train.predict_brep_edges(brep_graph_encoder, brep_graph_decoder, batch)

        # move tp device
        node_features = node_features.to(torch.float32).to(device).squeeze(0)
        operations_matrix = operations_matrix.to(torch.float32).to(device)
        intersection_matrix = intersection_matrix.to(torch.float32).to(device)
        operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)
        edge_features = edge_features.to(torch.float32).to(device).squeeze(0)


        gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
        brep_stroke_connection_matrix, stroke_coplanar_matrix = gnn_graph.set_brep_connection(edge_features, face_feature_gnn_list)

        stroke_weights = Models.sketch_model_helper.integrate_brep_probs(brep_edges_weights, brep_stroke_connection_matrix, stroke_coplanar_matrix)

        x_dict = brep_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
        output = strokes_decoder(x_dict, gnn_graph.edge_index_dict, stroke_weights)
        
        # Prepare gt
        target_op_index = len(program[0]) - 1
        op_to_index_matrix = operations_order_matrix
        kth_operation = Models.sketch_arguments.face_aggregate.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
        
        # Find the exact sketch
        sketch_strokes = Models.sketch_arguments.face_aggregate.face_aggregate_withMask(node_features, output)

        # Models.sketch_model_helper.vis_gt_strokes(node_features, kth_operation)
        # Models.sketch_model_helper.vis_gt_strokes(node_features, sketch_strokes)

        # Vis
        total_predictions +=1 
        gt_mask = (kth_operation > 0).float()
        pred_mask = (sketch_strokes > 0.2).float()

        if not torch.all(pred_mask == gt_mask):
        #     Models.sketch_model_helper.vis_stroke_cloud(node_features)
        #     Models.sketch_model_helper.vis_stroke_cloud(edge_features)
        #     Models.sketch_model_helper.vis_gt_strokes(node_features, kth_operation)
        #     Models.sketch_model_helper.vis_gt_strokes(node_features, output)
        
        #     Models.sketch_model_helper.vis_stroke_cloud(node_features)
        #     Models.sketch_model_helper.vis_stroke_cloud(edge_features)
        #     Models.sketch_model_helper.vis_gt_strokes(node_features, kth_operation)
        #     Models.sketch_model_helper.vis_gt_strokes(node_features, output)

            pass
        else:
            correct_predictions += 1

    # Compute the accuracy
    accuracy = correct_predictions / total_predictions

    # Print the accuracy
    print(f"Correct Predictions {correct_predictions} out of total {total_predictions}, Accuracy: {accuracy * 100:.2f}%")

#---------------------------------- Public Functions ----------------------------------#

# train()


