from torch.utils.data import Dataset
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 


class Program_Graph_Dataset(Dataset):
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), 'dataset')
        self.data_dirs = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
    
    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx):
        data_dir = self.data_dirs
        [idx]
        data_path = os.path.join(self.data_path, data_dir)
        print("data_path", data_path)
        return idx



dataset = Program_Graph_Dataset()

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

for batch in tqdm(data_loader):
    idx = batch
