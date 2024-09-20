from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader

class FeynmanDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
     
    def __len__(self):
        return len(self.dataset)
 
    def __getitem__(self, idx):
        item = self.dataset[idx]
     
        # Extract inputs and target
        inputs = torch.tensor([item[f'x{i}'] for i in range(1, 6) if f'x{i}' in item], dtype=torch.float32)
        target = torch.tensor([item['y']], dtype=torch.float32)
     
        # If you need variance, you might need to compute it or use a default value
        variance = torch.tensor([0.01], dtype=torch.float32)  # default value, adjust as needed
     
        return inputs, target, variance

def get_feynman_dataloader(batch_size=32):
    # Load the dataset
    dataset = load_dataset("yoshitomo-matsubara/srsd-feynman_medium")

    # The dataset typically has 'train', 'validation', and 'test' splits
    train_data = dataset['train']

    # Create dataset and dataloader
    train_dataset = FeynmanDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader
