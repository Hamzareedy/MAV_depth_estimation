import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import config
import random


class DepthDataset(Dataset):
    def __init__(self):
        self.image_path = config.config["image_path"]
        self.depth_path = config.config["depth_path"]
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, f"image_{idx:05d}.jpg")
        depth_path = os.path.join(self.depth_path, f"image_{idx:05d}.jpg")
        
        img = load_image(img_path) # 3 * H * W
        depth = load_image(depth_path, mode="L") # 1 * H * W, convert to grayscale maps
        
        return img, depth
        
    def __len__(self):
        # There's a hidden file called ".DS_store" (some mac thing) which means this method counts 1 too many images
        # if you don't do -1
        return len(os.listdir(self.image_path)) - 1
    
    
def load_image(path, mode = "RGB"):
    '''
        Load image from path and convert to tensor
    '''
    img = Image.open(path).convert(mode)
    return T.ToTensor()(img)


def load_train_val_dataset():
    '''
        Load dataset from config file
    '''
    dataset = DepthDataset()
    ratio = config.config["train_val_split"]
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [ratio, 1 - ratio])
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.config["batch_size"], 
        shuffle=True, 
        num_workers=config.config["num_workers"]
        )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.config["batch_size"], 
        shuffle=False, 
        num_workers=config.config["num_workers"]
        )
    return train_dataloader, val_dataloader

def load_eval_dataset(num_imgs):
    '''
        Load num_imgs number of images for evaluation
    '''
    dataset = DepthDataset()
    random_indices = random.sample(range(len(dataset)), num_imgs)
    eval_dataset = torch.utils.data.Subset(dataset, random_indices)

    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.config["num_workers"]
        )
    
    return eval_data_loader