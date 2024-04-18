import torch
from torch import nn
from torch.nn import functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
        self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        self.discriminator_output = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            # nn.Sigmoid()
        )


    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.discriminator_output(x)
        return x