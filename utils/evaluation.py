import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim 
import lpips 

def metrics(preds, targets, model_lpips = lpips.LPIPS(net='vgg')):
    val_psnr = psnr(preds, targets, 1.0)
    val_ssim = ssim(preds, targets, 1.0)


    preds_ = (preds*2)-1
    targets_ = targets*2 -1
    val_lpips = model_lpips(preds_, targets_)
    val_lpips = val_lpips.mean()
    
    return {
        "PSNR": val_psnr.item(),
        "SSIM": val_ssim.item(),
        "LPIPS": val_lpips.item()
    }
    
def evaluate_model(model, data_loader, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), loss_fn = nn.L1Loss()):
    psnr_loss = 0.0
    ssim_loss = 0.0
    lpips_loss = 0.0
    model_lpips = lpips.LPIPS(net='vgg').to(device)
    eval_bar = tqdm(data_loader, desc="Evaluating")
    model.to(device)
    for low_light, light_img in eval_bar:
        low_light = low_light.to(device)
        light_img = light_img.to(device)
        model.eval()
        with torch.no_grad():
            pred = model(low_light)
            metrics_ = metrics(pred, light_img, model_lpips)
            psnr_loss += metrics_['PSNR']
            ssim_loss += metrics_["SSIM"]
            lpips_loss += metrics_["LPIPS"]
    psnr_avg = psnr_loss / len(data_loader)
    ssim_avg = ssim_loss / len(data_loader)
    lpips_avg = lpips_loss / len(data_loader)
    print(f"PSNR LOSS: {psnr_avg} \n SSIM LOSS: {ssim_avg} \n LPIPS LOSS: {lpips_avg}") 

def unormalize(img):
    return (img+1.0)/2.0

def draw_images(model, data_loader, device = torch.cuda, num_images=5):
    model.eval()
    images_drawn = 0
    with torch.no_grad():
        for low_light, light_ in data_loader:
            low_light = low_light.to(device)
            light_ = light_.to(device)
            pred = model(low_light)
            for i in range(low_light.size(0)):
                if images_drawn >= num_images:
                    return  
                
                # low_img = unormalize(low_light[i])
                # pred_img = unormalize(pred[i])
                # light_img = unormalize(light_img[i])
                
                low_img = np.transpose(low_light[i].cpu().numpy(), (1,2,0))
                pred_img = np.transpose(pred[i].cpu().numpy(), (1,2,0))
                light_img = np.transpose(light_[i].cpu().numpy(), (1,2,0))
                
                low_img = np.clip(low_img, 0, 1)
                pred_img = np.clip(pred_img, 0, 1)
                light_img = np.clip(light_img, 0, 1)
                
                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1)
                plt.title('Low Light Image')
                plt.imshow(low_img)
                plt.axis('off')
                
                plt.subplot(1,3,2)
                plt.title('Enhanced Image')
                plt.imshow(pred_img)
                plt.axis('off')
                
                plt.subplot(1,3,3)
                plt.title('Ground Truth Image')
                plt.imshow(light_img)
                plt.axis('off')
                
                plt.show()
                images_drawn += 1
        