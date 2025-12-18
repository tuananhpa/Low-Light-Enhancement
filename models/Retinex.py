import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# Decomposition Network
# ------------------------
class DecomNet(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(channel, 4, 3, 1, 1)  # R(3) + L(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        out = self.conv4(x)
        R = torch.sigmoid(out[:, :3, :, :])
        L = torch.sigmoid(out[:, 3:, :, :])
        return R, L


# ------------------------
# Illumination Enhancement Network
# ------------------------
class EnhanceNet(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(channel, 1, 3, 1, 1)

    def forward(self, L):
        x = F.relu(self.conv1(L))
        x = F.relu(self.conv2(x))
        out = torch.sigmoid(self.conv3(x))
        return out


# ------------------------
# Retinex-Net Wrapper
# ------------------------
class RetinexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = DecomNet()
        self.enhance = EnhanceNet()

    def forward(self, x):
        R, L = self.decom(x)
        L_enhanced = self.enhance(L)
        output = R * L_enhanced
        return output, R, L, L_enhanced
