import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionTransformer(nn.Module):
    def __init__(self, nhead=8, num_encoder_layers=4, dim_feedforward=128, dropout=0.1):
        super(SelfAttentionTransformer, self).__init__()
        self.embedding_dim = 32
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.fc = nn.Linear(self.embedding_dim, 10) 
        
    def forward(self, graph_embedding, program_embedding, brep_embedding):
        # Pad the embeddings to the required shapes
        if graph_embedding.size(1) < 100:
            graph_embedding = F.pad(graph_embedding, (0, 0, 0, 100 - graph_embedding.size(1)), "constant", 0)
        if program_embedding.size(1) < 20:
            program_embedding = F.pad(program_embedding, (0, 0, 0, 20 - program_embedding.size(1)), "constant", 0)
        if brep_embedding.size(1) < 100:
            brep_embedding = F.pad(brep_embedding, (0, 0, 0, 100 - brep_embedding.size(1)), "constant", 0)
        
        # Combine the embeddings
        combined_embedding = torch.cat((graph_embedding, program_embedding, brep_embedding), dim=1)  # Shape: (1, 220, 32)
        
        # Apply transformer
        transformer_output = self.transformer_encoder(combined_embedding.permute(1, 0, 2))  # Shape: (220, 1, 32)
        
        # Permute back
        transformer_output = transformer_output.permute(1, 0, 2)  # Shape: (1, 220, 32)
        
        # Pooling
        pooled_output = torch.mean(transformer_output, dim=1)  # Shape: (1, 32)
        
        # Final classification layer
        logits = self.fc(pooled_output)  # Shape: (1, 10)
        
        return logits


    def predict_label(self, graph_embedding, program_embedding, brep_embedding):
        logits = self.forward(graph_embedding, program_embedding, brep_embedding)
        predicted_label = torch.argmax(logits, dim=1)
        return predicted_label
