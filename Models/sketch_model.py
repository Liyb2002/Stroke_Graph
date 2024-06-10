import torch
import torch.nn as nn

class GraphBrepAttention(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=128, dropout=0.1):
        super(GraphBrepAttention, self).__init__()
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

    def forward(self, graph_embedding, brep_embedding):
        graph_embedding = graph_embedding.transpose(0, 1)  # Shape: (n, 1, 32)
        brep_embedding = brep_embedding.transpose(0, 1)    # Shape: (m, 1, 32), where m is the number of brep embeddings
        
        # Cross attention
        attn_output, _ = self.cross_attn(graph_embedding, brep_embedding, brep_embedding)  # (n, 1, 32)
        attn_output = self.dropout(attn_output)
        out1 = self.norm1(graph_embedding + attn_output)  # (n, 1, 32)
        
        # Feed-forward
        ff_output = self.ff(out1)  # (n, 1, 32)
        out2 = self.norm2(out1 + ff_output)  # (n, 1, 32)
        
        # Classification
        logits = self.classifier(out2)  # (n, 1, 1)
        logits = logits.squeeze(-1)  # (n, 1)
        
        return logits
