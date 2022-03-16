from unet_parts import *


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.mid = conv_block(512, 1024)
        self.decoder = Decoder()
        self.segmap = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x.reverse()

        x[0] = self.mid(x[0])

        x = self.decoder(x)
        x = self.segmap(x)
        x = self.sigmoid(x)

        return x

if __name__ == "__main__":
    image = torch.rand((1, 3, 572, 572))
    model = UNet()
    print(model(image).shape)