import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


class conv_block(nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(insize, outsize, 3),
            nn.ReLU(),
            nn.Conv2d(outsize, outsize, 3),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = []

        conv1 = self.conv1(x)
        out.append(conv1)
        conv1 = self.maxpool(conv1)

        conv2 = self.conv2(conv1)
        out.append(conv2)
        conv2 = self.maxpool(conv2)

        conv3 = self.conv3(conv2)
        out.append(conv3)
        conv3 = self.maxpool(conv3)

        conv4 = self.conv4(conv3)
        out.append(conv4)
        conv4 = self.maxpool(conv4)
        out.append(conv4)

        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(1024, 512)
        self.conv2 = conv_block(512, 256)
        self.conv3 = conv_block(256, 128)
        self.conv4 = conv_block(128, 64)

        self.deconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 2, 2)

    def forward(self, enc_output):
        out = self.deconv1(enc_output[0])
        enc_output[1] = CenterCrop((out.shape[2], out.shape[3]))(enc_output[1])
        out = torch.cat((enc_output[1], out), dim = 1)
        out = self.conv1(out)

        out = self.deconv2(out)
        enc_output[2] = CenterCrop((out.shape[2], out.shape[3]))(enc_output[2])
        out = torch.cat((enc_output[2], out), dim = 1)
        out = self.conv2(out)

        out = self.deconv3(out)
        enc_output[3] = CenterCrop((out.shape[2], out.shape[3]))(enc_output[3])
        out = torch.cat((enc_output[3], out), dim = 1)
        out = self.conv3(out)

        out = self.deconv4(out)
        enc_output[4] = CenterCrop((out.shape[2], out.shape[3]))(enc_output[4])
        out = torch.cat((enc_output[4], out), dim = 1)
        out = self.conv4(out)

        return out