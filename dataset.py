import os
import numpy as np
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
        self.image_mode = config.config["image_mode"]
        self.in_type_uint8 = config.config["input_type_uint8"]
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, f"image_{idx:05d}.jpg")
        
        img = load_image(img_path, mode=self.image_mode, use_uint8=self.in_type_uint8) # 3 * H * W (rgb) or 1 * H * W (grayscale)
        # depth = load_image(depth_path, mode="L") # 1 * H * W, convert to grayscale maps
        depth_matrix = np.load(os.path.join(self.depth_path, f"array_{idx:05d}.npy"))
        depth_vector = extract_center_from_depthmatrix(depth_matrix) # 1 * H
        
        return img, depth_vector
        
    def __len__(self):
        # There's a hidden file called ".DS_store" (some mac thing) which means this method counts 1 more image
        # if you don't do -1
        return len(os.listdir(self.image_path)) - 1
    
    
def load_image(path, mode = "RGB", use_uint8=False):
    '''
        Load image from path and convert to tensor
    '''
    img = Image.open(path).convert(mode)
    if use_uint8: return T.PILToTensor()(img)
    return T.ToTensor()(img)


def extract_center_from_depthmatrix(depth_matrix):
    H, W = depth_matrix.shape
    center_depth = depth_matrix[:, W//2] # 1 * H
    return center_depth


def extract_center_from_depthmap(batch_depth_map):
    '''
        Extract the center line depth from the depth map
        
        ATTENTION: 
        Since the depth map is always rotated by 90 degrees, the center line is actually the center column
    '''
    _, _, _, W = batch_depth_map.shape
    center_depth = batch_depth_map[:, :, :, W//2] # N * 1 * H
    # Interpolate the depth to H/8
    downsampled_depth = F.avg_pool1d(center_depth, kernel_size=8, stride=8) # N * 1 * H/8
    downsampled_depth = downsampled_depth.squeeze(1) # N * H/8x
    # print(f"Extracted center depth shape: {downsampled_depth.shape}")
    return downsampled_depth


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