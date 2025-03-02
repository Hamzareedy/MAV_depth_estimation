import os
import time
import logging
import argparse
import h5py
from PIL import Image
import config


def parse_args():
    '''
        Parse arguments from command line
    '''
    parser = argparse.ArgumentParser(description="Depth Estimation")
    parser.add_argument("--mode", type=str, default=None, help="Mode: train/eval")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    args = parser.parse_args()
    return args


def init_logger():
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("Running on device: {}".format(config.config["device"]))
    
    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # File handler for file output
    current_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime).to_string()
    file_handler = logging.FileHandler(os.path.join(config.config["save_log_path"], f"log_{current_time}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


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
    pass