import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer


class StrokeEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=6, embedding_dim=32):
        super(StrokeEmbeddingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class PlaneEmbeddingNetwork(nn.Module):
    def __init__(self, stroke_embedding_dim=16, hidden_dim=32, output_dim=32):
        super(PlaneEmbeddingNetwork, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=stroke_embedding_dim, num_heads=1, batch_first=True)
        self.fc = nn.Linear(stroke_embedding_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, edge_index_face_edge_list, index_id, node_embed):
        face_to_edges = {}

        # if we have empty brep
        if node_embed.shape[1] == 1:
            return torch.zeros((1, 3, 32))

        # pair the edges index with each face
        for face_edge_pair in edge_index_face_edge_list:
            face_list_index = face_edge_pair[0]
            edge_list_index = face_edge_pair[1]

            face_id = index_id[face_list_index].item()
            edge_id = index_id[edge_list_index].item()

            if face_id not in face_to_edges:
                face_to_edges[face_id] = []
            face_to_edges[face_id].append(edge_id)
            
            

        face_embeddings = []
        for face_id, edge_ids in face_to_edges.items():
            
            # face_edges: shape (1, 4, 32)
            face_edges = node_embed[:, edge_ids, :]

            # attention_output: shape (1, 4, 32)
            attention_output, _ = self.self_attention(face_edges, face_edges, face_edges)
            
            # x: shape (1, 4, 32)
            x = self.relu(self.fc(attention_output))
            
            # x: shape (1, 32)
            x = x.mean(dim=1)
            
            # face_embedding: shape (1, 32)
            face_embedding = self.fc_output(x)
            face_embeddings.append(face_embedding)

        # Stack the embeddings for all faces to form the output tensor
        face_embeddings = torch.stack(face_embeddings, dim=1)

        # face_embedding: shape (1, num_faces, 32)
        return face_embeddings



class FaceBrepAttention(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=128, dropout=0.1):
        super(FaceBrepAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, face_embedding, brep_embedding):
        face_embedding = face_embedding.transpose(0, 1)  # Shape: (n, 1, 32)
        brep_embedding = brep_embedding.transpose(0, 1)    # Shape: (m, 1, 32), where m is the number of brep embeddings
        
        # Cross attention
        attn_output, _ = self.cross_attn(face_embedding, brep_embedding, brep_embedding)  # (n, 1, 32)
        attn_output = self.dropout(attn_output)
        out1 = self.norm1(face_embedding + attn_output)  # (n, 1, 32)
        
        # Feed-forward
        ff_output = self.ff(out1)  # (n, 1, 32)
        out2 = self.norm2(out1 + ff_output)  # (n, 1, 32)
        
        # Classification
        logits = self.classifier(out2)  # (n, 1, 1)
        logits = logits.squeeze(1)  # (n, 1)
        
        probabilities = torch.sigmoid(logits)
        return probabilities

    def predict(self, face_embedding, brep_embedding):
        logits = self.forward(face_embedding, brep_embedding)        
        
        probabilities = torch.sigmoid(logits)

        # Find the face with the highest probability above the threshold
        max_prob, max_index = torch.max(probabilities, dim=0)

        return max_index.item(), max_prob.item()




class Stroke_cross_attention_model(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=128, dropout=0.1):
        super(Stroke_cross_attention_model, self).__init__()
        self.cross_attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.final_layer = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, graph_embedding, face_embedding, brep_embedding):
        graph_embedding = graph_embedding.transpose(0, 1)
        face_embedding = face_embedding.transpose(0, 1)
        brep_embedding = brep_embedding.transpose(0, 1)
        
        attn_output1, _ = self.cross_attn1(graph_embedding, face_embedding, face_embedding)        
        attn_output1 = self.dropout(attn_output1)
        out1 = self.norm1(graph_embedding + attn_output1)
        
        attn_output2, _ = self.cross_attn2(out1, brep_embedding, brep_embedding)
        attn_output2 = self.dropout(attn_output2)
        out2 = self.norm2(out1 + attn_output2)
        
        ff_output = self.ff(out2)
        out2_final = self.final_layer(ff_output).squeeze(-1) 
        out2_final = self.sigmoid(out2_final)
        
        return out2_final



class DummyClassifier(nn.Module):
    def __init__(self, embedding_dim=32):
        super(DummyClassifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)
        
    def forward(self, x):
        # x shape: (1, num_faces, 32)
        x = x.squeeze(0)  # shape: (num_faces, 32)
        x = self.fc(x)  # shape: (num_faces, 1)
        x = torch.sigmoid(x)  # Ensure output is between 0 and 1
        return x



class BrepStrokeCloudAttention(nn.Module):
    def __init__(self, input_dim=128, num_heads=32, dropout=0.1):
        super(BrepStrokeCloudAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Dropout(dropout)
        )
        self.output_layer1 = nn.Linear(input_dim, 32)  # Output layer to compute scores for each edge
        self.output_layer2 = nn.Linear(32, 1)  # Output layer to compute scores for each edge

    def forward(self, brep_feature, stroke_cloud):
        # brep_feature: (1, n, 32)
        # stroke_cloud: (1, m, 32)
        brep_feature = brep_feature.permute(1, 0, 2)  # (n, 1, 32)
        stroke_cloud = stroke_cloud.permute(1, 0, 2)  # (m, 1, 32)
        
        attn_output, _ = self.attention(stroke_cloud, brep_feature, brep_feature)  # Cross attention
        attn_output = attn_output.permute(1, 0, 2)  # (1, n, 32)
        
        stroke_cloud = stroke_cloud.permute(1, 0, 2)  # Back to (1, n, 32)
        stroke_cloud = self.layer_norm1(stroke_cloud + attn_output)  # Add & Norm
        
        ff_output = self.feed_forward(stroke_cloud)
        ff_output = self.layer_norm2(stroke_cloud + ff_output)  # Add & Norm
        
        # Compute edge scores
        ff_output = ff_output.squeeze(0)
        edge_scores = self.output_layer1(ff_output)  # (n, 1)
        edge_scores = self.output_layer2(edge_scores)  # (n, 1)

        # Compute probabilities using sigmoid
        edge_probabilities = torch.sigmoid(edge_scores)  # (n, 1)
        
        return edge_probabilities




class BrepStrokeCloudAttention_Reverse(nn.Module):
    def __init__(self, input_dim=32, num_heads=4, dropout=0.1):
        super(BrepStrokeCloudAttention_Reverse, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Dropout(dropout)
        )
        self.output_layer = nn.Linear(input_dim, 1)  # Output layer to compute scores for each edge

    def forward(self, brep_feature, stroke_cloud):
        # brep_feature: (1, n, 32)
        # stroke_cloud: (1, m, 32)
        brep_feature = brep_feature.permute(1, 0, 2)  # (n, 1, 32)
        stroke_cloud = stroke_cloud.permute(1, 0, 2)  # (m, 1, 32)
        
        attn_output, _ = self.attention(brep_feature, stroke_cloud, stroke_cloud)  # Cross attention
        attn_output = attn_output.permute(1, 0, 2)  # (1, n, 32)
        
        brep_feature = brep_feature.permute(1, 0, 2)  # Back to (1, n, 32)
        brep_feature = self.layer_norm1(brep_feature + attn_output)  # Add & Norm
        
        ff_output = self.feed_forward(brep_feature)
        ff_output = self.layer_norm2(brep_feature + ff_output)  # Add & Norm
        
        # Compute edge scores
        ff_output = ff_output.squeeze(0)
        feature_scores = self.output_layer(ff_output)  # (n, 1)
        
        # Compute probabilities using sigmoid
        feature_probabilities = torch.sigmoid(feature_scores)  # (n, 1)
        
        return feature_probabilities
