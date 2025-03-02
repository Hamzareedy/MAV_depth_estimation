import os
import time
import logging
import argparse
import config


def parse_args():
    '''
        Parse arguments from command line
    '''
    parser = argparse.ArgumentParser(description="Depth Estimation")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train/eval")
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
    current_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime)
    file_handler = logging.FileHandler(os.path.join(config.config["save_log_path"], f"log_{current_time}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger