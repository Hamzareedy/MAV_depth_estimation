# Configurations & Hyperparameters for model training
import torch

config = {
    # Model training configurations
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 100,
    "batch_size": 64,
    "train_test_split": 0.8,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "num_workers": 4,
    "ssim_weight": 0.8
    
    # Model configurations
    "input_channels": 4,
    "output_channels": 1,
    
    # Data paths
    "train_data_path": "data/train",
    # "val_data_path": "data/val",
    "test_data_path": "data/test",
    
    # Model paths
    "save_model_path": "models",
    "save_log_path": "logs",
}