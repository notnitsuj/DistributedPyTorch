from .unet_parts import *


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.mid = conv_block(256, 512)
        self.decoder = Decoder()
        self.segmap = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, conv1, conv2, conv3, conv4 = self.encoder(x)
        x = self.mid(x)
        x = self.decoder(x, conv1, conv2, conv3, conv4)
        x = self.segmap(x)
        x = self.sigmoid(x)

        return x

if __name__ == "__main__":
    image = torch.rand((1, 3, 640, 960))
    model = UNet()
    print(model(image).shape)