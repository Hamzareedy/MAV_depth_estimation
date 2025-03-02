import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import config, utils, model, dataset

def train():
    # Set up logging & device
    logger = utils.init_logger()
    device = config.config["device"]
    
    # Load dataset
    train_loader = dataset.load_dataset(config.config["train_data_path"])
    
    # Load model
    model = model.DepthEstimation()
    model.to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    
    # Loss function & optimizer
    weight = config.config["ssim_weight"]
    loss = lambda x, y: weight * (1 - nn.SSIM()(x, y)) + (1 - weight) * nn.MSELoss()(x, y)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.config["learning_rate"], 
        weight_decay=config.config["weight_decay"]
        )
    
    # Training loop
    writer = SummaryWriter("logs")
    for epoch in range(config.config["epochs"]):
        for i, (img, depth) in enumerate(train_loader):
            img, depth = img.to(device), depth.to(device)
            rgbd = torch.cat([img, depth], dim=1)
            
            optimizer.zero_grad()
            
            pred_depth = model(rgbd)
            loss_val = loss(pred_depth, depth)
            loss_val.backward()
            optimizer.step()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch}, Iteration {i}, Loss: {loss_val.item()}")
                writer.add_scalar("Loss/train", loss_val.item(), epoch * len(train_loader) + i)
        
        torch.save(model.state_dict(), os.path.join(config.config["save_model_path"], f"model_{epoch}.pth"))
        
    logger.info("Training complete.")
    writer.close()
    
    
    
if __name__ == "__main__":
    args = utils.parse_args()
    if args.mode == "train":
        train()
    elif args.mode == "eval":
        pass
    else:
        # utils.convert_h5_to_image('data/depth_maps_cyberzoo_aggressive_flight_20190121-144646.h5')
        utils.rename_files("data/20190121-144646")
        # raise ValueError("Invalid mode. Please choose 'train' or 'eval'.")
    


