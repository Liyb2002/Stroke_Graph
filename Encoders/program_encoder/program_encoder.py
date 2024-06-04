import torch
import torch.nn as nn

class ProgramEncoder(nn.Module):
    def __init__(self, vocab_size=10, embedding_dim=8, hidden_dim=16, num_layers=1):
        super(ProgramEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, program_data):
        # program_data: (batch_size, seq_length)
        program_embeddings = self.embedding(program_data)
        # program_embeddings: (batch_size, seq_length, embedding_dim)
        _, (hidden_state, _) = self.lstm(program_embeddings)
        # hidden_state: (num_layers, batch_size, hidden_dim)
        # Take the hidden state of the last LSTM layer
        program_encoding = hidden_state[-1]
        # program_encoding: (batch_size, hidden_dim)
        return program_encoding

