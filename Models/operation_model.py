import torch
import torch.nn as nn

class CrossAttentionTransformer(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=128, num_classes=10, dropout=0.1):
        super(CrossAttentionTransformer, self).__init__()
        self.cross_attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, graph_embedding, program_encoding, brep_embedding):
        graph_embedding = graph_embedding.transpose(0, 1)
        program_encoding = program_encoding.transpose(0, 1)
        brep_embedding = brep_embedding.transpose(0, 1)
        
        attn_output1, _ = self.cross_attn1(brep_embedding, graph_embedding, graph_embedding)        
        attn_output1 = self.dropout(attn_output1)
        out1 = self.norm1(brep_embedding + attn_output1)
        
        attn_output2, _ = self.cross_attn2(program_encoding, out1, out1)
        attn_output2 = self.dropout(attn_output2)
        out2 = self.norm2(program_encoding + attn_output2)
        
        ff_output = self.ff(out2)        
        out3_mean = ff_output.mean(dim=0)
        
        logits = self.classifier(out3_mean)
        
        return logits

    def predict_label(self, graph_embedding, program_encoding, brep_embedding):
        logits = self.forward(graph_embedding, program_encoding, brep_embedding)
        predicted_label = torch.argmax(logits)
        return predicted_label
