import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(ResBlock, self).__init__()

        self.downsample = downsample
        if self.downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + shortcut
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes):
        super(ResNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # original: 7,2,3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer2 = self.make_layer(ResBlock, in_channels=64, out_channels=64, num_blocks=2, downsample=False)
        self.layer3 = self.make_layer(ResBlock, in_channels=64, out_channels=128, num_blocks=2, downsample=True)
        self.layer4 = self.make_layer(ResBlock, in_channels=128, out_channels=256, num_blocks=2, downsample=True)
        self.layer5 = self.make_layer(ResBlock, in_channels=256, out_channels=512, num_blocks=2, downsample=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    def make_layer(self, ResBlock, in_channels, out_channels, num_blocks, downsample):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, downsample))
        for _ in range(num_blocks-1):
            layers.append(ResBlock(out_channels, out_channels, downsample=False))
        return nn.Sequential(*layers)

def Res18():
    return ResNet(ResBlock, num_classes=10)
