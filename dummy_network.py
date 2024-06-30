import torch
import torch.nn as nn

class CrossAttentionNetwork(nn.Module):
    def __init__(self, input_size=3, num_heads=1, dropout=0.1):
        super(CrossAttentionNetwork, self).__init__()
        self.embedding_layer1 = nn.Linear(input_size, input_size)  # Embedding layer for tensor1
        self.embedding_layer2 = nn.Linear(input_size, input_size)  # Embedding layer for tensor2
        self.multihead_attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout)
        self.linear = nn.Linear(input_size, 1)  # Linear layer for final output
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation

    def forward(self, tensor1, tensor2):
        # Embedding tensors separately
        embedded_tensor1 = self.embedding_layer1(tensor1)  # Shape: (seq_len1, input_size)
        embedded_tensor2 = self.embedding_layer1(tensor2)  # Shape: (seq_len2, input_size)
        
        # print("boundary_points_matrix", embedded_tensor1)
        # print("vertex points", embedded_tensor2)
        # Compute multihead attention
        attn_output, _ = self.multihead_attention(query=embedded_tensor1.unsqueeze(1),
                                                  key=embedded_tensor2.unsqueeze(1),
                                                  value=embedded_tensor2.unsqueeze(1))
        
        # Linear layer to get logits (no activation)
        logits = self.linear(attn_output.squeeze(1))  # Shape: (seq_len2, 1)
        
        # Apply sigmoid activation to get probabilities
        probabilities = self.sigmoid(logits)  # Shape: (seq_len2, 1)
        
        return probabilities.squeeze(1)  # Shape: (seq_len2,)
