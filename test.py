import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class MultiheadAttentionNetwork(nn.Module):
    def __init__(self, input_dim=32, num_heads=8, dropout=0.1):
        super(MultiheadAttentionNetwork, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Dropout(dropout)
        )
        self.output_layer = nn.Linear(input_dim, 1)  # Output layer to compute scores for each edge
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensorA, tensorC):
        # Apply Multihead Attention
        attn_output, _ = self.attention(tensorC, tensorA, tensorA)
        
        # Apply layer normalization and residual connection
        norm_output = self.layer_norm1(attn_output + tensorC)
        
        # Apply feed-forward network
        ff_output = self.feed_forward(norm_output)
        
        # Apply second layer normalization and residual connection
        norm_ff_output = self.layer_norm2(ff_output + norm_output)
        
        # Apply output layer and sigmoid activation
        output = self.sigmoid(self.output_layer(norm_ff_output))
        
        return output.squeeze(-1)  # Shape (batch_size,)


def generate_data():
    num_embedding = 32
    # Create random tensors
    tensorA_base = torch.randn(4, 8)  # Shape (4, 6)
    tensorB_base = torch.randn(4, 8)  # Shape (4, 6)

    # Concatenate tensors A and B along the first dimension
    tensorC_base = torch.cat([tensorA_base, tensorB_base], dim=0)  # Shape (8, 6)

    permuted_indices = torch.randperm(8)
    tensorC_base = tensorC_base[permuted_indices]

    # Determine which indices in tensorC correspond to tensorA
    indices_in_A = torch.zeros(8, dtype=torch.float)  # Initialize a tensor to store results


    for i in range(8):
        # Check if each row of tensorC is in tensorA
        for j in range(4):
            if torch.equal(tensorC_base[i], tensorA_base[j]):
                indices_in_A[i] = 1
                break  # Found a match, no need to check further


    # Now add noise
    additional_data_A = torch.randn(4, num_embedding - tensorA_base.size(1))  # Shape (4, 26)
    tensorA = torch.cat([tensorA_base, additional_data_A], dim=1)  # Shape (4, num_embedding)
    additional_data_A2 = torch.randn(8, num_embedding)  # Shape (8, num_embedding)
    tensorA = torch.cat([tensorA, additional_data_A2], dim=0)   # Shape (12, num_embedding)

    # Generate random data to expand tensorC_base to (8, num_embedding)
    additional_data_C = torch.randn(8, num_embedding - tensorC_base.size(1))  # Shape (8, 26)
    tensorC = torch.cat([tensorC_base, additional_data_C], dim=1)  # Shape (8, num_embedding)



    
    return tensorA, tensorC, indices_in_A


def create_dataset(n):
    dataset = []
    for _ in range(n):
        tensorA, tensorC, indices_in_A = generate_data()
        dataset.append((tensorA, tensorC, indices_in_A))
    return dataset


train_dataset = create_dataset(10000)  # Training dataset with 10000 samples
val_dataset = create_dataset(1000)    # Validation dataset with 1000 samples

# Initialize model, criterion, and optimizer
model = MultiheadAttentionNetwork()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss with logits
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer

# Training parameters
num_epochs = 100

# Training loop with tqdm for progress visualization
for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}/{num_epochs}")
    
    # Training phase
    model.train()
    train_losses = []
    with tqdm(total=len(train_dataset), unit="batch") as progress_bar:
        for tensorA, tensorC, indices_in_A in train_dataset:
            # Forward pass
            logits = model(tensorA, tensorC)
            
            # Calculate loss
            loss = criterion(logits, indices_in_A)
            train_losses.append(loss.item())
            
            # print("--------")
            # print("tensorA", tensorA.shape)
            # print("tensorB", tensorC.shape)

            # print("tensorA", tensorA)
            # print("tensorB", tensorC)

            # print("indices_in_A", indices_in_A)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update tqdm progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({"train_loss": loss.item()})
    
    # Average training loss
    avg_train_loss = sum(train_losses) / len(train_losses)
    
    # Validation phase
    model.eval()
    val_losses = []
    with torch.no_grad():
        for tensorA, tensorC, indices_in_A in val_dataset:
            logits = model(tensorA, tensorC)
            val_loss = criterion(logits, indices_in_A)
            val_losses.append(val_loss.item())
    
    # Average validation loss
    avg_val_loss = sum(val_losses) / len(val_losses)
    
    # Print epoch results
    print(f"Epoch {epoch} Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

print("Training finished.")
