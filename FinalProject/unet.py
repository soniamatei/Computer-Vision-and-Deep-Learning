import torch
import torch.nn as nn
import torchvision.transforms.functional as fn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        """
        A class consisting of two 3x3 convolutions which preserve the input dimensions. Each
        is followed by Batch norm. and a ReLU.
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # * stride of 1 when preserving spatial dimensions of the input
            # * id stride = 1 we do not take into consideration in_size of the image for the padding
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        """
        Takes the forward pass in the module.
        @return: the result of the convolutions
        """
        return self.conv(x)


class TransposeConvolution(nn.Module):
    def __init__(self, in_channels: int):
        """
        A class consisting of one transpose convolution followed by Batch norm. and a ReLU.
        """
        super(TransposeConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        """
        Takes the forward pass in the module.
        @return: the result of the convolutions
        """
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, features: list[int], kernel_size: int):
        """
        Used as a part for U-net, builds a fully convolutional module which squishes the data given.
        Uses as many double convolutions as the length of feature list, each followed by a ReLU function.
        Max pooling with size=2 and stride=2 for pooling.
        @param in_channels: the dimension which will be given to the first convolution
        @param features: - a list of dimensions which will be the output for each double conv. respectively
                         - must be in increasing order
        """
        super(Encoder, self).__init__()
        self.downs = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(2, 2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, kernel_size))
            in_channels = feature

        self.downs.append(DoubleConv(features[-1], features[-1] * 2, kernel_size))

    def forward(self, x:  torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Takes the forward pass in the module. Between convolutions, a skip connection is taken.
        @param x: batch input of shape [batch size, nr. channels, width, height]
        @return: the resulting output from last convolutions and the skip connections(arranged
                from the first conv. to the last)
        """
        skip_connections = []

        for down in self.downs[:-1]:
            x = down(x)
            skip_connections.append(x)
            x = self.max_pool(x)

        x = self.downs[-1](x)

        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self, out_channels: int, features: list[int], kernel_size: int):
        """
        Used as a part for U-net, builds a fully convolutional module which enlarges the data given.
        Uses as many double convolutions as the length of feature list, each followed by a ReLU function.
        Transposed convolutions with size=2 and stride=2 for unpooling.
        @param out_channels: the dimension which will be given to the last convolution
        @param features: - a list of dimensions which will be the output for each double conv. respectively
                         - must be in decreasing order
        """
        super(Decoder, self).__init__()
        self.ups = nn.ModuleList()

        for feature in features:
            self.ups.append(TransposeConvolution(feature))
            self.ups.append(DoubleConv(feature * 2, feature, kernel_size))

        self.ups.append(nn.Conv2d(features[-1], out_channels, 1))

    def forward(self, x:  torch.Tensor, skip_connections: torch.Tensor) -> torch.Tensor:
        """
        Takes the forward pass in the module.
        @param x: batch input of shape [batch size, nr. channels, width, height]
        @param skip_connections: required list of tensors to be concatenated at each step in the required order after
                                the number of channels (descending for the decoder)
        @return: the result obtained
        """
        for i in range(0, len(self.ups) - 1, 2):
            conv_trans = self.ups[i]
            conv_double = self.ups[i + 1]

            x = conv_trans(x)
            skip_connection_cropped = fn.center_crop(skip_connections[i//2], [x.shape[2], x.shape[3]])
            # * concatenate along the channel dimension
            x = torch.cat((skip_connection_cropped, x), 1)
            x = conv_double(x)

        # obtain the segmentation map
        x = self.ups[-1](x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: list[int], kernel_size: int):
        """
        Unet module composed of two parts: encoder and decoder.
        @param in_channels: the number of channels of each image in the batch
        @param out_channels: the number of classes each pixel will be classified to
        @param features: the depth needed for the steps in the encoder/decoder
        @param kernel_size: the width and height of the kernel
        """
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, features, kernel_size)
        self.decoder = Decoder(out_channels, features[::-1], kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip_connections = self.encoder.forward(x)
        segmentation_map = self.decoder(x, skip_connections[::-1])

        return segmentation_map

    @staticmethod
    def true_segmentation(prediction: torch.Tensor) -> torch.Tensor:
        """
        Helper function for creating and RGB only segmentation from a predicted one which may not
        have absolute values.
        @param prediction: predicted segmentation of shape 3xwxh
        @return: wxhx3 segmentation
        """
        max_channel_indices = torch.argmax(prediction, dim=0)

        r_mask = (max_channel_indices == 0).float()
        g_mask = (max_channel_indices == 1).float()
        b_mask = (max_channel_indices == 2).float()

        seg_map_rgb = torch.stack([r_mask, g_mask, b_mask], axis=2)

        return seg_map_rgb

# unet = UNet(in_channels=3, out_channels=3, kernel_size=3, features=[64, 128, 256, 512])
# inn = random_integers = torch.randint(0, 10, (1, 3, 224, 224)).float()
# print(inn.shape)
# print(unet(inn).shape)
