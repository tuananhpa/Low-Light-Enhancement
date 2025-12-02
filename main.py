from utils.dataset import get_loader
from models.modules import Generator
from turtle import forward
from torch._dynamo.convert_frame import input_codes
from utils.dataset import get_loader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim
from tqdm import tqdm
import os

if __name__ == "__main__":
    root_dir = './dataset'
    if not os.path.exists(root_dir):
        print("Dataset not found")
        exit()
    mode = 'train'
    batch_size = 16
    num_workers = 4
    train_loader = get_loader(root_dir, mode, batch_size, num_workers)
    valid_loader = get_loader(root_dir, 'val', batch_size, num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    model = Generator()
    model.to(device)
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)
    criteria = nn.L1Loss()
    
  
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for low_img, light_img in train_loader:
            low_img = low_img.to(device)
            light_img = light_img.to(device)
            optimizer.zero_grad()
            pred = model(low_img)
            loss = criteria(pred, light_img)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        model.eval()
        avg_train_loss = train_loss / len(train_loader)
        valid_loss = 0
        with torch.no_grad():
            val_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            for low_img, light_img in val_bar:
                low_img = low_img.to(device)
                light_img = light_img.to(device)
                pred = model(low_img)
                loss = criteria(pred, light_img)
                valid_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
        avg_valid_loss = valid_loss / len(valid_loader)