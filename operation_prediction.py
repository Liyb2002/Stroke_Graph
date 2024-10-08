import Preprocessing.dataloader
import Preprocessing.gnn_graph_full
import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network
import Models.sketch_model_helper

import Encoders.gnn_full.gnn

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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define the neural networks

graph_encoder = Encoders.gnn_full.gnn.SemanticModule()
graph_decoder = Encoders.gnn_full.gnn.Program_prediction()

graph_encoder.to(device)
graph_decoder.to(device)

# Directory for saving models
current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'operation_prediction')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    # Load models if they exist
    if os.path.exists(os.path.join(save_dir, 'graph_encoder.pth')):
        graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth')))
        print("Loaded graph_encoder")

    if os.path.exists(os.path.join(save_dir, 'graph_decoder.pth')):
        graph_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth')))
        print("Loaded graph_decoder")


def save_models():
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))
    torch.save(graph_decoder.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))
    print("Saved models.")

# Define optimizer and loss function
optimizer = optim.Adam( list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)
loss_function = nn.CrossEntropyLoss()


def load_dataset():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/extrude_only_test')

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_loader, val_loader


def load_eval_dataset():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/extrude_only_eval')
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    return eval_loader


def train():
    train_loader, val_loader = load_dataset()
    # Training and validation loop
    best_val_loss = float('inf')
    epochs = 30

    for epoch in range(epochs):
        # Training
        graph_encoder.train()
        graph_decoder.train()
        
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, edge_features, brep_coplanar, new_features= batch
        
            node_features = node_features.to(torch.float32).to(device).squeeze(0)
            edge_features = torch.tensor(edge_features, dtype=torch.float32)

            # Now build graph
            gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.set_brep_connection(edge_features)

            # program embedding
            gt_next_token = program[0][-1]
            current_program = program[0][:-1]

            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict, current_program)

            # Forward pass through cross attention model
            loss = loss_function(output, gt_next_token)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()


        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
    
        # Validation
        graph_encoder.eval()
        graph_decoder.eval()
        
        total_val_loss = 0.0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, edge_features, brep_coplanar, new_features= batch
            
                node_features = node_features.to(torch.float32).to(device).squeeze(0)
                edge_features = torch.tensor(edge_features, dtype=torch.float32)

                # Now build graph
                gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
                gnn_graph.set_brep_connection(edge_features)

                # program embedding
                gt_next_token = program[0][-1]
                current_program = program[0][:-1]

                x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
                output = graph_decoder(x_dict, current_program)

                # Forward pass through cross attention model
                loss = loss_function(output, gt_next_token)
                total_val_loss += loss.item()

                # Calculate accuracy
                _, predicted_class = torch.max(output, dim = 0)
                correct_predictions += (predicted_class == gt_next_token).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct_predictions / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, Correct Predictions: {correct_predictions}/{len(val_loader)}")

        # Save model if validation loss decreases
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_models()


def eval():
    load_models()
    eval_loader = load_eval_dataset()

    graph_encoder.eval()
    graph_decoder.eval()

    total_eval_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, edge_features, brep_coplanar, new_features= batch
        
            node_features = node_features.to(torch.float32).to(device).squeeze(0)
            edge_features = torch.tensor(edge_features, dtype=torch.float32)

            # Now build graph
            gnn_graph = Preprocessing.gnn_graph_full.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.set_brep_connection(edge_features)

            # program embedding
            gt_next_token = program[0][-1]
            current_program = program[0][:-1]

            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict, current_program)

            # Forward pass through cross attention model
            loss = loss_function(output, gt_next_token)
            total_eval_loss += loss.item()

            if gt_next_token == 0:
                Models.sketch_model_helper.vis_compare(node_features, edge_features)
                # print("all present", Models.sketch_model_helper.all_stroke_represented(x_dict['stroke']))

            # Predictions and labels for confusion matrix
            _, predicted_class = torch.max(output, 0)  # Use dim=0 if output is a single-dimensional tensor
            all_preds.append(predicted_class.item())
            all_labels.append(gt_next_token.item())

    avg_eval_loss = total_eval_loss / len(eval_loader)
    print(f"Evaluation Loss: {avg_eval_loss:.4f}")

    # Compute and display confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("conf_matrix")
    print(conf_matrix)
#---------------------------------- Public Functions ----------------------------------#

eval()
