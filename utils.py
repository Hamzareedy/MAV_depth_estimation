import os
import time
import logging
import argparse
import h5py
import random
from PIL import Image
from matplotlib import pyplot as plt
import config
import numpy as np


def parse_args():
    '''
        Parse arguments from command line
    '''
    parser = argparse.ArgumentParser(description="Depth Estimation")
    parser.add_argument("--mode", type=str, default=None, help="Mode: data/train/eval")
    parser.add_argument("--path", type=str, default="data/depth_maps_cyberzoo_aggressive_flight_20190121-144646.h5", help="h5 file path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--model_id", type=int, default=0, help="Model ID for evaluation")
    args = parser.parse_args()
    return args


def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # File handler for file output
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    file_handler = logging.FileHandler(os.path.join(config.config["save_log_path"], f"log_{current_time}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # log device info
    for i, item in enumerate(config.config.items()):
        if i == 8: break
        logger.info(item.__str__())
    
    return logger


def convert_h5_to_array(h5_path): 
    '''
        Load depth matrix from h5 file and store as numpy 
    '''
    output_path = os.path.join("data", "depth_matrix")
    os.makedirs(output_path, exist_ok=True)
    
    with h5py.File(h5_path, "r") as f:
        for i, key in enumerate(f.keys()):
            # 520 * 240 array store the depth information
            array = f[f[key].name][:]
            # print(f"Array : {array}")
            # print(f"Array shape: {array.shape}")
            # break
            array_path = os.path.join(output_path, f"array_{i:05d}.npy")
            np.save(array_path, array)
            print(f"Save {f[key].name} array to {array_path}")


def convert_h5_to_image(h5_path): 
    '''
        Load image from h5 file and store in 
    '''
    output_path = os.path.join("data", "depth_map")
    os.makedirs(output_path, exist_ok=True)
    
    with h5py.File(h5_path, "r") as f:
        # print(f.keys())
        for i, key in enumerate(f.keys()):
            img_array = f[f[key].name][:]
            img = Image.fromarray(img_array)
            img_path = os.path.join(output_path, f"image_{i:05d}.jpg")
            img.save(img_path)
            print(f"Save {f[key].name} image to {img_path}")
    
    
def rename_files(path):
    '''
        Rename files in a folder
    '''
    files = sorted(os.listdir(path))
    output_path = os.path.join("data", "original_image")
    print(f"Found {len(files)} files")
    
    for i, file in enumerate(files):
        os.rename(os.path.join(path, file), os.path.join(output_path, f"image_{i:05d}.jpg"))
        print(f"Rename {file} to image_{i:05d}.jpg")
     

def load_comparison():
    '''
        Randomly load a pair of image and depth map for comparison
    '''
    img_data_folder = config.config["image_path"]
    depth_data_folder = config.config["depth_path"]
    
    idx = random.randint(0, len(os.listdir(img_data_folder)))
    img = Image.open(os.path.join(img_data_folder, f"image_{idx:05d}.jpg"))
    depth = Image.open(os.path.join(depth_data_folder, f"image_{idx:05d}.jpg"))
    
    # Rotate the image and depth map by 90 degrees
    img_rotated = img.rotate(90, expand=True)
    depth_rotated = depth.rotate(90, expand=True)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rotated)
    plt.title(f"original image {idx} (rotated)")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(depth_rotated, cmap="gray")
    plt.title(f"depth map {idx} (rotated)")
    plt.axis("off")
    plt.show()
    
    
def show_eval_images(depth_pred, img, depth_gt):
    '''
        Display the input image, ground truth and predicted depth maps
    '''
    if config.config["image_mode"] == "RGB": img = np.transpose(img, (1, 2, 0))
    img_rotated = np.rot90(img, k=1)
    depth_pred_rotated = np.rot90(depth_pred, k=1)
    depth_gt_rotated = np.rot90(depth_gt, k=1)

    plt.figure()
    plt.subplot(3, 1, 1)
    if config.config["image_mode"] == "RGB": plt.imshow(img_rotated, cmap="gray")
    else: plt.imshow(img_rotated)
    plt.title(f"Original image")
    plt.axis("off")
    
    plt.subplot(3, 1, 2)
    plt.imshow(depth_gt_rotated, cmap="gray")
    plt.title(f"Depth ground truth")
    plt.axis("off")

    plt.subplot(3, 1, 3)
    plt.imshow(depth_pred_rotated, cmap="gray")
    plt.title(f"Predicted depth map")
    plt.axis("off")
    plt.show()