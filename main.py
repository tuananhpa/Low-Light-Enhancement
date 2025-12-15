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
import joblib
import gc

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
    if not os.path.exists('./checkpoint'):
        os.makedirs('./checkpoint')
    gc.collect()
    torch.cuda.empty_cache()
    start_epoch = 1
    try: 
        last_checkpoint = sorted(os.listdir('./checkpoint'))[-1]
        model.load_state_dict(torch.load(os.path.join('./checkpoint', last_checkpoint)))
        start_epoch = int(last_checkpoint.split('_')[-1].split('.')[0])
        print(f"Loaded checkpoint: {last_checkpoint}")
        print(f"Resuming training from epoch {start_epoch}")
    except:
        print("No checkpoint found, training from scratch.")
    for epoch in range(epochs-start_epoch):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+start_epoch+1}/{epochs} [Train]")
        for low_img, light_img in train_bar:
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
        torch.save(model.state_dict(), f'./checkpoint/generator_epoch_{epoch+start_epoch+1}.pth')
        with torch.no_grad():
            val_bar = tqdm(valid_loader, desc=f"Epoch {epoch+start_epoch+1}/{epochs} [Valid]")
            for low_img, light_img in val_bar:
                low_img = low_img.to(device)
                light_img = light_img.to(device)
                pred = model(low_img)
                loss = criteria(pred, light_img)
                valid_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
        avg_valid_loss = valid_loss / len(valid_loader)
        

        