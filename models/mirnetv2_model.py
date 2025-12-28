
import torch
import torch.nn as nn


def conv(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)


# ------------------ Channel Attention ------------------
class CALayer(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.conv(w)
        return x * w


# ------------------ Residual Block ------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            conv(ch, ch),
            nn.ReLU(inplace=True),
            conv(ch, ch)
        )
        self.ca = CALayer(ch)

    def forward(self, x):
        res = self.block(x)
        res = self.ca(res)
        return x + res


# ------------------ Down / Up ------------------
class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.down = nn.Sequential(
            conv(ch, ch, 3, 2, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.up = nn.Sequential(
            conv(ch, ch * 4, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


# ------------------ Selective Kernel Feature Fusion ------------------
class SKFF(nn.Module):
    def __init__(self, ch, branches=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // 8, 1),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([nn.Conv2d(ch // 8, ch, 1) for _ in range(branches)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feats):
        stack = torch.stack(feats, dim=1)
        fea_sum = torch.sum(stack, dim=1)
        att = self.pool(fea_sum)
        att = self.fc(att)
        att = torch.stack([fc(att) for fc in self.fcs], dim=1)
        att = self.softmax(att)
        return torch.sum(stack * att, dim=1)


# ------------------ Multi-Resolution Block ------------------
class MRB(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.rb1 = ResBlock(ch)
        self.rb2 = ResBlock(ch)
        self.rb3 = ResBlock(ch)

        self.down1 = Down(ch)
        self.down2 = Down(ch)
        self.up1 = Up(ch)
        self.up2 = Up(ch)

        self.fuse1 = SKFF(ch, 2)
        self.fuse2 = SKFF(ch, 3)
        self.fuse3 = SKFF(ch, 2)

    def forward(self, x1, x2, x3):
        x1 = self.rb1(x1)
        x2 = self.rb2(x2)
        x3 = self.rb3(x3)

        f1 = self.fuse1([x1, self.up1(x2)])
        f2 = self.fuse2([self.down1(x1), x2, self.up2(x3)])
        f3 = self.fuse3([self.down2(x2), x3])

        return f1, f2, f3


# ------------------ MIRNet v2 ------------------
class MIRNetV2(nn.Module):
    def __init__(self, ch=64):
        super().__init__()

        self.head = conv(3, ch)

        self.down1 = Down(ch)
        self.down2 = Down(ch)

        self.rrg1 = MRB(ch)
        self.rrg2 = MRB(ch)
        self.rrg3 = MRB(ch)

        self.up1 = Up(ch)
        self.up2 = Up(ch)

        self.fuse = SKFF(ch, 3)

        self.tail = nn.Sequential(
            conv(ch, ch),
            nn.ReLU(inplace=True),
            conv(ch, 3)
        )

    def forward(self, x):
        s1 = self.head(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)

        s1, s2, s3 = self.rrg1(s1, s2, s3)
        s1, s2, s3 = self.rrg2(s1, s2, s3)
        s1, s2, s3 = self.rrg3(s1, s2, s3)

        s2 = self.up1(s2)
        s3 = self.up1(self.up2(s3))

        out = self.fuse([s1, s2, s3])
        out = self.tail(out)

        return torch.clamp(out + x, 0, 1)
