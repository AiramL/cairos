import torch
import torch.nn as nn

class FlisbeeNet(nn.Module):
    """
    CNN simples de 3 camadas para o servidor.
    - Conv → BN → ReLU → Pool (x2) e uma terceira conv sem pool grande,
      depois FC -> logits.
    - Dimensões calculadas para CIFAR-10 (3x32x32). Ajuste se mudar resolução.
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        # Conv1: 32x32 -> 16x16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Conv2: 16x16 -> 8x8
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Conv3: 8x8 -> 8x8 (sem pool grande)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Cabeça
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # (B,64,16,16)
        x = self.conv2(x)  # (B,128,8,8)
        x = self.conv3(x)  # (B,256,8,8)
        x = self.avgpool(x)  # (B,256,1,1)
        x = torch.flatten(x, 1)
        x = self.fc(x)  # logits
        return x