import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import config


class DepthDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.image_path = os.path.join(root_path, "images")
        self.depth_path = os.path.join(root_path, "depth")
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, f"{idx}.jpg")
        depth_path = os.path.join(self.depth_path, f"{idx}.png")
        
        img = load_image(img_path) # 3 * H * W
        depth = load_image(depth_path) # 1 * H * W
        
        return img, depth
        
    def __len__(self):
        return len(os.listdir(self.image_path))
    
    
def load_image(path):
    '''
        Load image from path and convert to tensor
    '''
    img = Image.open(path).convert("RGB")
    return F.to_tensor(img)


def load_dataset(path):
    '''
        Load dataset from config file
    '''
    dataset = DepthDataset(path)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.config["batch_size"], 
        shuffle=True, 
        num_workers=config.config["num_workers"]
        )
    return dataloader