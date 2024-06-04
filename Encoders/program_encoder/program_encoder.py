import torch
import torch.nn as nn

class ProgramEncoder(nn.Module):
    def __init__(self, vocab_size=10, embedding_dim=8, hidden_dim=16, num_layers=1):
        super(ProgramEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, program_data):
        program_indices = self.convert_to_indices(program_data)
        # program_data: (batch_size, seq_length)
        program_embeddings = self.embedding(program_indices)
        # program_embeddings: (batch_size, seq_length, embedding_dim)
        _, (hidden_state, _) = self.lstm(program_embeddings)
        # hidden_state: (num_layers, batch_size, hidden_dim)
        # Take the hidden state of the last LSTM layer
        program_encoding = hidden_state[-1]
        # program_encoding: (batch_size, hidden_dim)
        return program_encoding

    def convert_to_indices(self, program_data):
        # Create a mapping from operations to indices
        operation_to_index = {'sketch': 0, 'extrude': 1, 'fillet': 2}  # Add more operations as needed
        
        # Convert program_data to a list of indices
        program_indices = []
        for program in program_data:
            for op in program:
                print("op", op)
            indices = [operation_to_index[op[0]] for op in program]
            program_indices.append(indices)
        
        # Convert program_indices to a tensor
        program_indices = torch.tensor(program_indices, dtype=torch.long)
        
        return program_indices
