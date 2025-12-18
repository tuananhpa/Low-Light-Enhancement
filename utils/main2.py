from utils.dataset import get_loader
from models.Retinex import RetinexNet
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import gc

if __name__ == "__main__":
    root_dir = "./dataset"
    batch_size = 16
    epochs = 10
    num_workers = 4

    train_loader = get_loader(root_dir, "train", batch_size, num_workers)
    val_loader = get_loader(root_dir, "val", batch_size, num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RetinexNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    l1_loss = nn.L1Loss()

    if not os.path.exists("./checkpoints/Retinex"):
        os.makedirs("./checkpoints/Retinex")

    gc.collect()
    torch.cuda.empty_cache()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for low_img, high_img in train_bar:
            low_img = low_img.to(device)
            high_img = high_img.to(device)

            optimizer.zero_grad()
            output, R, L, L_enhanced = model(low_img)

            recon_loss = l1_loss(output, high_img)
            illum_smooth = torch.mean(torch.abs(L[:, :, :, :-1] - L[:, :, :, 1:])) + \
                           torch.mean(torch.abs(L[:, :, :-1, :] - L[:, :, 1:, :]))

            loss = recon_loss + 0.1 * illum_smooth
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        torch.save(
            model.state_dict(),
            f"./checkpoints/Retinex/retinex_epoch_{epoch+1}.pth"
        )
