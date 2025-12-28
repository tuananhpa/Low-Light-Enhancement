
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from models.model_mirnetv2 import MIRNetV2
from utils.dataset import get_loader


def save_checkpoint(root, epoch, model, optimizer):
    gen_dir = os.path.join(root, "Gen")
    dis_dir = os.path.join(root, "Dis")

    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(dis_dir, exist_ok=True)

    path = os.path.join(gen_dir, f"generator_epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path
    )

    dummy = os.path.join(dis_dir, "README.txt")
    if not os.path.exists(dummy):
        with open(dummy, "w") as f:
            f.write("Placeholder for discriminator folder.")

    return path


def validate(model, loader, device, criterion):
    model.eval()
    total = 0
    count = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                low, high = batch
            else:
                low, high = batch[0], batch[1]

            low = low.to(device)
            high = high.to(device)

            pred = model(low)
            loss = criterion(pred, high)

            total += loss.item()
            count += 1

    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--ckpt_root", type=str, default="checkpoints/MIRNetv2")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train_loader = get_loader(args.data_root, args.batch_size, mode="train")
        val_loader = get_loader(args.data_root, args.batch_size, mode="val")
    except TypeError:
        train_loader = get_loader(args.data_root, args.batch_size)
        val_loader = None

    model = MIRNetV2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    start_epoch = 1
    best_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0
        count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            if len(batch) == 2:
                low, high = batch
            else:
                low, high = batch[0], batch[1]

            low = low.to(device)
            high = high.to(device)

            pred = model(low)
            loss = criterion(pred, high)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1