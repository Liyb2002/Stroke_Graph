import torch

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
