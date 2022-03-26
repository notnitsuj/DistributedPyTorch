import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


class conv_block(nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(insize, outsize, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outsize, outsize, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3, 32)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64, 128)
        self.conv4 = conv_block(128, 256)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.maxpool(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.maxpool(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.maxpool(conv3)

        conv4 = self.conv4(pool3)
        out = self.maxpool(conv4)
        
        return out, conv4, conv3, conv2, conv1
      
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(512, 256)
        self.conv2 = conv_block(256, 128)
        self.conv3 = conv_block(128, 64)
        self.conv4 = conv_block(64, 32)

        self.deconv1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 2, 2)

    def forward(self, x, conv1, conv2, conv3, conv4):
        out = self.deconv1(x)
        conv1 = CenterCrop((out.shape[2], out.shape[3]))(conv1)
        out = torch.cat((conv1, out), dim = 1)
        out = self.conv1(out)

        out = self.deconv2(out)
        conv2 = CenterCrop((out.shape[2], out.shape[3]))(conv2)
        out = torch.cat((conv2, out), dim = 1)
        out = self.conv2(out)

        out = self.deconv3(out)
        conv3 = CenterCrop((out.shape[2], out.shape[3]))(conv3)
        out = torch.cat((conv3, out), dim = 1)
        out = self.conv3(out)

        out = self.deconv4(out)
        conv4 = CenterCrop((out.shape[2], out.shape[3]))(conv4)
        out = torch.cat((conv4, out), dim = 1)
        out = self.conv4(out)

        return out