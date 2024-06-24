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
