import torch
from torch import nn
from training.segformer import MiT
from torch.nn import functional as F

class lp3d(nn.Module):
    def __init__(self,
                 pretrained_backbone: bool = False):
        super().__init__()
        model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet50', pretrained=pretrained_backbone)
        self.deeplabv3 = model.backbone
        self.conv1 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1)
        self.vit1 = encF()
        self.encoder_high = encH()
        self.vit2 = encT()

    def forward(self, x):
        feat = self.deeplabv3(x)['out']
        F_low = self.conv1(feat)
        f = self.vit1(F_low)
        F_high = self.encoder_high(x)
        x = self.vit2(f, F_high)
        return x
    
class encH(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(self, x):
        return self.block0(x)
    
class encF(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = MiT(
            channels = 256,
            dims = 1024,
            heads = 4,
            ff_expansion = 2,
            reduction_ratio = 1,
            num_layers = 5
        )
        self.pixshuf = nn.PixelShuffle(2)
        self.seq1 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )
        self.seq2 = nn.Sequential( 
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.pixshuf(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.seq1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.seq2(x)
        return x
        
class encT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.vit = MiT(
            channels = 128,
            dims = 1024,
            heads = 4,
            ff_expansion = 4,
            reduction_ratio = 1,
            num_layers = 1
        )
        self.pixshuf = nn.PixelShuffle(2)
        self.block1 = nn.Sequential(
            nn.Conv2d(352, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.block0(x)
        x = self.vit(x)
        x = self.pixshuf(x)
        x = torch.cat([x, y], dim=1)
        x = self.block1(x)
        return x