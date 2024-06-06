import torch
import torch.nn as nn


class CrossAttentionTransformer(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, num_layers=4):
        super(CrossAttentionTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 10)

    def forward(self, graph1_embed, string_embed, graph2_embed):
        string_embed_expanded = string_embed.unsqueeze(1).expand(1, -1, -1)

        combined = torch.cat((graph1_embed, string_embed_expanded, graph2_embed), dim=1)
        attn_output = self.transformer_encoder(combined)
        attn_output = attn_output.mean(dim=1)
        out = self.fc(attn_output)
        return out
