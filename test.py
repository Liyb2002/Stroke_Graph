import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class MultiheadAttentionNetwork(nn.Module):
    def __init__(self, input_dim=128, num_heads=16, dropout=0.1):
        super(MultiheadAttentionNetwork, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
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
    # Create random tensors
    tensorA = torch.randn(4, 128)  # Shape (4, 1)
    tensorB = torch.randn(4, 128)  # Shape (4, 1)

    # Concatenate tensors A and B along the first dimension
    tensorC = torch.cat([tensorA, tensorB], dim=0)  # Shape (8, 1)

    permuted_indices = torch.randperm(8)
    tensorC = tensorC[permuted_indices]

    # Determine which indices in tensorC correspond to tensorA
    indices_in_A = torch.zeros(8, dtype=torch.float)  # Initialize a tensor to store results

    for i in range(8):
        # Check if each row of tensorC is in tensorA
        for j in range(4):
            if torch.equal(tensorC[i], tensorA[j]):
                indices_in_A[i] = 1
                break  # Found a match, no need to check further
    
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
num_epochs = 10

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
