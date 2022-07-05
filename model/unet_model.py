from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, pipe=False):
        super().__init__()
        self.encoder = Encoder()
        self.mid = conv_block(256, 512)
        self.decoder = Decoder()
        self.segmap = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.pipeline = pipe

        if self.pipeline:
            self.encoder.to('cuda:0')
            self.mid.to('cuda:0')
            #self.seq2 = nn.Sequential(self.decoder, self.segmap, self.sigmoid).to('cuda:1')
            self.decoder.to('cuda:1')
            self.segmap.to('cuda:1')
            self.sigmoid.to('cuda:1')
            print(self.decoder)

    def forward(self, x):
        if self.pipeline:
            split_size = int(x.size(0) / 2)

            splits = iter(x.split(split_size, dim=0))
            s_next = next(splits)
            s_prev, conv1, conv2, conv3, conv4 = self.encoder(s_next)
            s_prev = self.mid(s_prev)
            ret = []

            for s_next in splits:
                # A. s_prev runs on cuda:1
                #s_prev = self.seq2(s_prev, conv1, conv2, conv3, conv4)
                s_prev = self.decoder(s_prev.to('cuda:1'), conv1.to('cuda:1'), conv2.to('cuda:1'), 
                                    conv3.to('cuda:1'), conv4.to('cuda:1'))
                s_prev = self.segmap(s_prev)
                s_prev = self.sigmoid(s_prev)
                ret.append(s_prev)

                # B. s_next runs on cuda:0, which can run concurrently with A
                s_prev, conv1, conv2, conv3, conv4 = self.encoder(s_next)
                s_prev = self.mid(s_prev).to('cuda:1')

            #s_prev = self.seq2(s_prev, conv1, conv2, conv3, conv4)
            s_prev = self.decoder(s_prev.to('cuda:1'), conv1.to('cuda:1'), conv2.to('cuda:1'), 
                                conv3.to('cuda:1'), conv4.to('cuda:1'))
            s_prev = self.segmap(s_prev)
            s_prev = self.sigmoid(s_prev)
            ret.append(s_prev)

            out = torch.cat(ret).to('cuda:0')

        else:
            x, conv1, conv2, conv3, conv4 = self.encoder(x)
            x = self.mid(x)
            x = self.decoder(x, conv1, conv2, conv3, conv4)
            x = self.segmap(x)
            out = self.sigmoid(x)

        return out

if __name__ == "__main__":
    image = torch.rand((1, 3, 640, 960))
    model = UNet()
    print(model(image).shape)