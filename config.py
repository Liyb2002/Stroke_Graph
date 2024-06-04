import torch


# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------- #

operation_to_index = {'sketch': 0, 'extrude_addition': 1, 'extrude_subtraction': 2, 'fillet': 3}