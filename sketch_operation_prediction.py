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


def plot_strokes_3d(strokes, stroke_features):
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
            target_op_index = len(program)
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
                target_op_index = len(program)
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
                plot_strokes_3d(gt_matrix, gnn_graph['stroke'].x)
                break

    avg_eval_loss = total_eval_loss / len(eval_loader)
    print(f"Evaluation Loss: {avg_eval_loss}")





#---------------------------------- Public Functions ----------------------------------#

eval()
