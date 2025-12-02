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

class MSFEM(nn.Module):
    def __init__(self, input_size = 64, output_size = 64) -> None:
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels= input_size, out_channels= input_size, kernel_size = 1, stride = 1, padding = 0)
        self.branch2 = nn.Conv2d(in_channels= input_size, out_channels= input_size, kernel_size = 3, stride = 1, padding = 1)
        self.branch3 = nn.Conv2d(in_channels= input_size, out_channels= input_size, kernel_size = 5, stride = 1, padding = 2)
        self.Relu = nn.ReLU(inplace = True)
        self.comb = nn.Conv2d(in_channels = 3 * input_size, out_channels = output_size, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        x1 = self.Relu(self.branch1(x))
        x2 = self.Relu(self.branch1(x))
        x3 = self.Relu(self.branch1(x))
        combination = torch.cat([x1,x2,x3], dim = 1)
        x4 = self.comb(combination)
        return x4
            
class IlluminationAttention(nn.Module):
    def __init__(self, input_size = 64) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, 1),
            nn.ReLU(True),
            
            nn.Conv2d(input_size // 2, input_size, 3, 1, 1),
            nn.Sigmoid()
        )    

    def forward(self, x):
        attn_map = self.attn(x)
        out = x * attn_map
        return x + out
        
        
class Encoder(nn.Module):
    def __init__(self, input_size = 3, output_size = 32, kernel_size = 3, stride = 1, padding = 1, use_maxpool = True, use_module = True) -> None:
        super().__init__()

        arr_layer = []
        if use_maxpool:
            arr_layer.append(nn.MaxPool2d(2))
        arr_layer.append(nn.Conv2d(in_channels=input_size, out_channels= output_size, kernel_size= kernel_size, stride = stride, padding = padding))
        arr_layer.append(nn.LeakyReLU())
        arr_layer.append(nn.BatchNorm2d(output_size))
        arr_layer.append(nn.Conv2d(in_channels=output_size, out_channels= output_size, kernel_size= kernel_size, stride = stride, padding = padding))
        arr_layer.append(nn.LeakyReLU())
        arr_layer.append(nn.BatchNorm2d(output_size))
        
        if use_module:
            arr_layer.append(MSFEM(output_size, output_size))
            arr_layer.append(IlluminationAttention(output_size))
        
        self.block = nn.Sequential(*arr_layer)
    
    def forward(self, x):
        return self.block(x)
      
class Decoder(nn.Module):
    def __init__(self, input_size = 3, output_size = 32, kernel_size = 3, stride = 1, padding = 1, use_deconv = True, use_tanh = True, use_conv = True) -> None:
        super().__init__()
        arr_layer = []
        input_deconv = input_size
        if use_conv:
            arr_layer.append(nn.Conv2d(in_channels=input_size, out_channels= input_size // 2, kernel_size= kernel_size, stride = stride, padding = padding))
            arr_layer.append(nn.LeakyReLU())
            arr_layer.append(nn.BatchNorm2d(input_size // 2))
            arr_layer.append(nn.Conv2d(in_channels=input_size // 2, out_channels= output_size, kernel_size= kernel_size, stride = stride, padding = padding))
            arr_layer.append(nn.LeakyReLU())
            arr_layer.append(nn.BatchNorm2d(output_size))
            input_deconv = output_size
        if use_deconv:
            arr_layer.append(nn.ConvTranspose2d(input_deconv, output_size,3,2,1))
        if use_tanh:
            arr_layer.append(nn.Conv2d(input_deconv, output_size, kernel_size= kernel_size, stride = stride, padding = padding))
            arr_layer.append(nn.Tanh())
        self.block = nn.Sequential(*arr_layer)
        
    def forward(self, x, skip_connection = None):
        res = self.block(x)
        
        if skip_connection is not None:
            if res.shape != skip_connection.shape:
                res = F.interpolate(res, skip_connection.shape[2:])
            res = torch.concat([res, skip_connection], dim = 1)
        
        return res

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = Encoder(3,32, use_maxpool=False)
        self.encoder2 = Encoder(32,64)
        self.encoder3 = Encoder(64,128)
        self.encoder4 = Encoder(128,256)
        self.encoder5 = Encoder(256,512, use_module=False)
        self.bottemneck = Decoder(512,256, use_conv=False, use_tanh=False)
        self.decoder4 = Decoder(512,128, use_tanh=False)
        self.decoder3 = Decoder(256,64, use_tanh=False)
        self.decoder2 = Decoder(128,32, use_tanh= False)
        self.decoder1 = Decoder(64,3, use_deconv=False)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        btn = self.bottemneck(e5, e4)
        d4 = self.decoder4(btn, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2)
        
        return d1
        

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
    optimizer = torch.optim.Adam()
    criteria = nn.L1Loss()
    
  
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for low_img, light_img in train_loader:
            low_img.to(device)
            light_img.to(device)
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
                low_img.to(device)
                light_img.to(device)
                pred = model(low_img)
                loss = criteria(pred, light_img)
                valid_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
        avg_valid_loss = valid_loss / len(valid_loader)
