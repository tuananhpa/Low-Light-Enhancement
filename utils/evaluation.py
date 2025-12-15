import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_model(model, data_loader, device = torch.cuda, loss_fn = nn.L1Loss()):
    total_loss = 0.0
    eval_bar = tqdm(data_loader, desc="Evaluating")
    model.to(device)
    for low_light, light_img in eval_bar:
        low_light = low_light.to(device)
        light_img = light_img.to(device)
        model.eval()
        with torch.no_grad():
            pred = model(low_light)
            loss = loss_fn(pred, light_img)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def draw_images(model, data_loader, device = torch.cuda, num_images=5):
    model.eval()
    images_drawn = 0
    with torch.no_grad():
        for low_light, light_img in data_loader:
            low_light = low_light.to(device)
            pred = model(low_light)
            for i in range(low_light.size(0)):
                if images_drawn >= num_images:
                    return
                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1)
                plt.title('Low Light Image')
                plt.imshow(np.transpose(low_light[i].cpu().numpy(), (1,2,0)))
                plt.axis('off')
                
                plt.subplot(1,3,2)
                plt.title('Enhanced Image')
                plt.imshow(np.transpose(pred[i].cpu().numpy(), (1,2,0)))
                plt.axis('off')
                
                plt.subplot(1,3,3)
                plt.title('Ground Truth Image')
                plt.imshow(np.transpose(light_img[i].cpu().numpy(), (1,2,0)))
                plt.axis('off')
                
                plt.show()
                images_drawn += 1
        