import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer


class StrokeEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=6, embedding_dim=16):
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
        self.self_attention = nn.MultiheadAttention(embed_dim=stroke_embedding_dim, num_heads=2, batch_first=True)
        self.fc = nn.Linear(stroke_embedding_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, edge_index_face_edge_list, index_id, node_embed):
        face_to_edges = {}

        if node_embed.shape[1] == 1:
                return torch.zeros((1, 1, 32))

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
            # Extract the corresponding node embeddings for the edges of the current face

            face_edges = node_embed[:, edge_ids, :]

            # Self-attention mechanism
            attention_output, _ = self.self_attention(face_edges, face_edges, face_edges)
            
            # Apply the first fully connected layer
            x = self.relu(self.fc(attention_output))
            
            # Aggregate the embeddings (e.g., by mean)
            x = x.mean(dim=1)
            
            # Apply the output fully connected layer
            face_embedding = self.fc_output(x)
            face_embeddings.append(face_embedding)

        # Stack the embeddings for all faces to form the output tensor
        face_embeddings = torch.stack(face_embeddings, dim=1)

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
        logits = logits.squeeze(-1)  # (n, 1)
        
        return logits

    def predict(self, face_embedding, brep_embedding):
        logits = self.forward(face_embedding, brep_embedding)        
        
        probabilities = torch.sigmoid(logits)

        # Find the face with the highest probability above the threshold
        max_prob, max_index = torch.max(probabilities, dim=0)

        return max_index.item(), max_prob.item()
