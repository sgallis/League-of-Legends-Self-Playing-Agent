import torch
from torch.utils.data import Dataset


class BufferDataset(Dataset):
    def __init__(self, buffer):
        super(BufferDataset, self).__init__()
        self.buffer = buffer
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        self.obs.append(obs)
        self.actions.append(action)
        self.action_coords.append(action_coords)
        self.logps.append(logp)
        self.rewards.append(reward)
        return super().__getitem__(index)
