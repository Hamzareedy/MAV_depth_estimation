# Configurations & Hyperparameters for model training
import torch

config = {
    # Model training configurations
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "num_workers": 4,
    
    # Model configurations
    "input_channels": 4,
    "output_channels": 1,
    
    # Data paths
    "train_data_path": "data/train",
    "val_data_path": "data/val",
    "test_data_path": "data/test",
    
    # Model paths
    "save_model_path": "models",
    "save_log_path": "logs",
}