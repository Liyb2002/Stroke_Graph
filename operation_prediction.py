import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network

import Encoders.gnn.gnn
import Encoders.program_encoder.program_encoder

import Models.operation_model

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Define the neural networks
graph_embedding_model = Encoders.gnn.gnn.SemanticModule()
program_embedding_model = Encoders.program_encoder.program_encoder.ProgramEncoder()
SBGCN_model = Preprocessing.SBGCN.SBGCN_network.FaceEdgeVertexGCN()
cross_attention_model = Models.operation_model.CrossAttentionTransformer()

graph_embedding_model.to(device)
program_embedding_model.to(device)
SBGCN_model.to(device)
cross_attention_model.to(device)

# Directory for saving models
current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'operation_prediction')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    # Load models if they exist
    if os.path.exists(os.path.join(save_dir, 'graph_embedding_model.pth')):
        graph_embedding_model.load_state_dict(torch.load(os.path.join(save_dir, 'graph_embedding_model.pth')))
        print("Loaded graph_embedding_model")

    if os.path.exists(os.path.join(save_dir, 'program_embedding_model.pth')):
        program_embedding_model.load_state_dict(torch.load(os.path.join(save_dir, 'program_embedding_model.pth')))
        print("Loaded program_embedding_model")

    if os.path.exists(os.path.join(save_dir, 'SBGCN_model.pth')):
        SBGCN_model.load_state_dict(torch.load(os.path.join(save_dir, 'SBGCN_model.pth')))
        print("Loaded SBGCN_model")

    if os.path.exists(os.path.join(save_dir, 'cross_attention_model.pth')):    
        cross_attention_model.load_state_dict(torch.load(os.path.join(save_dir, 'cross_attention_model.pth')))
        print("Loaded cross_attention_model")


def save_models():
    torch.save(graph_embedding_model.state_dict(), os.path.join(save_dir, 'graph_embedding_model.pth'))
    torch.save(program_embedding_model.state_dict(), os.path.join(save_dir, 'program_embedding_model.pth'))
    torch.save(SBGCN_model.state_dict(), os.path.join(save_dir, 'SBGCN_model.pth'))
    torch.save(cross_attention_model.state_dict(), os.path.join(save_dir, 'cross_attention_model.pth'))
    print("Saved models.")


def vis_brep(brep_edge_features):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_edges = brep_edge_features.shape[1]

    for i in range(num_edges):
        start_point = brep_edge_features[0, i, :3]
        end_point = brep_edge_features[0, i, 3:]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def vis_next_step(node_features, operations_order_matrix, next_program_count):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    node_features = node_features.squeeze(0)
    operations_order_matrix = operations_order_matrix.squeeze(0)

    # Get the ith column from the operations_order_matrix
    if next_program_count >= operations_order_matrix.shape[1]:
        new_operations_order_matrix = torch.zeros((operations_order_matrix.shape[0], 1))
    else:
        new_operations_order_matrix = operations_order_matrix[:, next_program_count].reshape(-1, 1)

    # Plot strokes with color based on new_operations_order_matrix
    for i, stroke in enumerate(node_features):
        start = stroke[:3].numpy()
        end = stroke[3:].numpy()
        color = 'red' if new_operations_order_matrix[i] == 1 else 'blue'
        
        # Plot the line segment for the stroke
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], marker='o', color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def train():

    # Define training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(graph_embedding_model.parameters()) + 
        list(program_embedding_model.parameters()) + 
        list(SBGCN_model.parameters()) +
        list(cross_attention_model.parameters()), 
        lr=0.0005
    )

    epochs = 20

    # Create a DataLoader
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/train_dataset')

    # Split dataset into training and validation
    train_size = int(0.08 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        graph_embedding_model.train()
        program_embedding_model.train()
        SBGCN_model.train()
        cross_attention_model.train()
        
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, _, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

            # to device 
            node_features = node_features.to(torch.float32).to(device)
            operations_matrix = operations_matrix.to(torch.float32).to(device)
            intersection_matrix = intersection_matrix.to(torch.float32).to(device)
            operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)

            # graph embedding
            gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.to_device(device)
            graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)

            # program embedding + brep embedding
            gt_next_token = program[0][-1]
            current_program = program[0][:-1]

            if len(current_program) == 0:
                # is empty program
                program_encoding = torch.zeros(1, 1, 32, device=device)
                brep_embedding = torch.zeros(1, 1, 32, device=device)
            else:
                # is not empty program 
                # program embedding
                program_encoding = program_embedding_model(current_program)
                program_encoding = program_encoding.unsqueeze(0)
                
                # brep embedding
                brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
                            edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
                brep_graph.to_device(device)
                face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)
                brep_embedding = torch.cat((face_embedding, edge_embedding, vertex_embedding), dim=1)

            # Forward pass through cross attention model
            output = cross_attention_model(graph_embedding, program_encoding, brep_embedding)
            loss = criterion(output, gt_next_token.unsqueeze(0))

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation
        graph_embedding_model.eval()
        program_embedding_model.eval()
        SBGCN_model.eval()
        cross_attention_model.eval()

        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, _, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

                # to device 
                node_features = node_features.to(torch.float32).to(device)
                operations_matrix = operations_matrix.to(torch.float32).to(device)
                intersection_matrix = intersection_matrix.to(torch.float32).to(device)
                operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)

                # graph embedding
                gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
                gnn_graph.to_device(device)
                graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)

                # program embedding + brep embedding
                gt_next_token = program[0][-1]
                current_program = program[0][:-1]

                if len(current_program) == 0:
                    # is empty program
                    program_encoding = torch.zeros(1, 1, 32, device=device)
                    brep_embedding = torch.zeros(1, 1, 32, device=device)
                else:
                    # is not empty program 
                    # program embedding
                    program_encoding = program_embedding_model(current_program)
                    program_encoding = program_encoding.unsqueeze(0)
                    
                    # brep embedding
                    brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
                                edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
                    brep_graph.to_device(device)
                    face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)
                    brep_embedding = torch.cat((face_embedding, edge_embedding, vertex_embedding), dim=1)

                # Forward pass through cross attention model
                output = cross_attention_model(graph_embedding, program_encoding, brep_embedding)
                loss = criterion(output, gt_next_token.unsqueeze(0))

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

        # Save the model if validation loss has decreased
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_models()



def eval():
    # Load models
    load_models()
    
    # Set models to evaluation mode
    graph_embedding_model.eval()
    program_embedding_model.eval()
    SBGCN_model.eval()
    cross_attention_model.eval()

    # Create a DataLoader
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/eval_dataset')

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            node_features, operations_matrix, intersection_matrix, operations_order_matrix, _, program, _, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch

            # to device 
            node_features = node_features.to(torch.float32).to(device)
            operations_matrix = operations_matrix.to(torch.float32).to(device)
            intersection_matrix = intersection_matrix.to(torch.float32).to(device)
            operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)

            # graph embedding
            gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(node_features, operations_matrix, intersection_matrix, operations_order_matrix)
            gnn_graph.to_device(device)
            graph_embedding = graph_embedding_model(gnn_graph.x_dict, gnn_graph.edge_index_dict)

            # program embedding + brep embedding
            gt_next_token = program[0][-1]
            current_program = program[0][:-1]

            if len(current_program) == 0:
                # is empty program
                program_encoding = torch.zeros(1, 1, 32, device=device)
                brep_embedding = torch.zeros(1, 1, 32, device=device)
            else:
                # is not empty program 
                # program embedding
                program_encoding = program_embedding_model(current_program)
                program_encoding = program_encoding.unsqueeze(0)
                
                # brep embedding
                brep_graph = Preprocessing.SBGCN.SBGCN_graph.GraphHeteroData(face_features, edge_features, vertex_features, 
                            edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id)
                brep_graph.to_device(device)
                face_embedding, edge_embedding, vertex_embedding = SBGCN_model(brep_graph)
                brep_embedding = torch.cat((face_embedding, edge_embedding, vertex_embedding), dim=1)

            # Make prediction
            prediction = cross_attention_model.predict_label(graph_embedding, program_encoding, brep_embedding)
            predictions.append(prediction.item())
            ground_truths.append(gt_next_token.item())

            if prediction.item() != gt_next_token.item() and gt_next_token.item() == 1:
                print("prediction", prediction.item())
                print("gt", gt_next_token.item())
                vis_brep(edge_features)
                vis_next_step(node_features, operations_order_matrix, len(current_program) )

            # print(f"Prediction: {prediction.item()}, Ground Truth: {gt_next_token.item()}")

    # Calculate evaluation metrics
    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions, average='weighted')
    recall = recall_score(ground_truths, predictions, average='weighted')
    f1 = f1_score(ground_truths, predictions, average='weighted')
    conf_matrix = confusion_matrix(ground_truths, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate metrics for each class
    report = classification_report(ground_truths, predictions, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()

    # Save metrics to CSV
    pwd = os.getcwd()
    metrics_csv_path = os.path.join(pwd, 'dataset', 'classification_report.csv')
    metrics_df.to_csv(metrics_csv_path, index=True)

    print(f"Classification report saved to {metrics_csv_path}")

    return

#---------------------------------- Public Functions ----------------------------------#

# train()
eval()