import torch
import torch.nn as nn

class ProgramEncoder(nn.Module):
    def __init__(self, vocab_size=10, embedding_dim=8, hidden_dim=32):
        super(ProgramEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        encoded = lstm_out[:, -1, :]
        return encoded
