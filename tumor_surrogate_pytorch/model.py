import torch
from torch import nn

from tumor_surrogate_pytorch.attention_utils.grid_attention import GridAttentionBlock3D
from tumor_surrogate_pytorch.attention_utils.utils import init_weights

import matplotlib.pyplot as plt
class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ConvLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, act_func='relu'):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        if act_func == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = None

    def forward(self, x):
        if self.act is None:
            return self.conv(x)
        else:
            return self.act(self.conv(x))


class ManyfoldConvBlock3D(nn.Module):

    def __init__(self, layers, shortcut, skip_pos):
        super(ManyfoldConvBlock3D, self).__init__()
        self.skip_pos = skip_pos
        self.layers = nn.ModuleList(layers)
        self.shortcut = shortcut

    def forward(self, x, skip_x=None):
        lateral = None
        if skip_x is None:  # encoder
            skip_x = self.shortcut(x)
        else: # decoder
            skip_x = self.shortcut(skip_x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.skip_pos:
                x = x + skip_x
                lateral = x
        return x, lateral


class TumorSurrogate(nn.Module):
    def __init__(self, widths, n_cells, strides):
        super().__init__()
        input_channel = 128
        first_conv = ConvLayer3D(
            3, input_channel, kernel_size=3, stride=1,
        )

        self.encoder_blocks = [first_conv]
        for width, n_cell, s in zip(widths, n_cells, strides):
            conv_layers = []
            shortcut = IdentityLayer()
            if s == 1:
                skip_pos = n_cell - 1
            else:
                skip_pos = n_cell - 2
            for i in range(n_cell):
                if i == n_cell - 1:  # last layer of block is pooling or stride conv
                    stride = s
                else:
                    stride = 1
                conv_op = ConvLayer3D(in_channels=input_channel, out_channels=width, kernel_size=3, stride=stride)
                conv_layers.append(conv_op)
                input_channel = width

            conv_block = ManyfoldConvBlock3D(conv_layers, shortcut, skip_pos=skip_pos)
            self.encoder_blocks.append(conv_block)

        mid_conv = ConvLayer3D(
            input_channel, input_channel - 3, kernel_size=3, stride=1
        )
        self.encoder_blocks.append(mid_conv)

        self.decoder_blocks = []
        n_cells_decoder = [x + 1 for x in n_cells]
        for width, n_cell, s in zip(widths, n_cells_decoder, strides):
            conv_layers = []
            if s == 1:
                skip_pos = n_cell - 1
            else:
                skip_pos = n_cell - 2
            shortcut = IdentityLayer()
            for i in range(n_cell):
                if i == n_cell - 1 and s != 1:  # last layer of block is Upsampling
                    conv_op = nn.Upsample(scale_factor=s, mode='nearest')
                else:
                    conv_op = ConvLayer3D(in_channels=input_channel, out_channels=width, kernel_size=3, stride=1)

                conv_layers.append(conv_op)
                input_channel = width
            conv_block = ManyfoldConvBlock3D(conv_layers, shortcut, skip_pos=skip_pos)
            self.decoder_blocks.append(conv_block)
            if s != 1:
                after_upscale_conv = ConvLayer3D(
                    in_channels=2*input_channel, out_channels=width,
                    kernel_size=3, stride=1
                )
                self.decoder_blocks.append(after_upscale_conv)
        # final layer
        last_channel = 1
        last_conv = ConvLayer3D(
            input_channel, last_channel, kernel_size=3, stride=1, act_func = None
        )
        self.decoder_blocks.append(last_conv)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)
        self.parameter_encoder = nn.Linear(in_features=3, out_features=3*8*8*8)

        self.attentionblock1 = MultiAttentionBlock(in_size=128, gate_size=128, inter_size=128,
                                              nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2))
        self.attentionblock2 = MultiAttentionBlock(in_size=128, gate_size=128, inter_size=128,
                                              nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2))
        self.attentionblock3 = MultiAttentionBlock(in_size=128, gate_size=128, inter_size=128,
                                              nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2))

        self.dsv2 = UnetDsv3(in_size=128, out_size=1, scale_factor=4)
        self.dsv1 = UnetDsv3(in_size=128, out_size=1, scale_factor=2)
        self.final = nn.Conv3d(in_channels=3, out_channels=1, kernel_size=1)


    def forward(self, x, parameters):
        # First Conv
        x = self.encoder_blocks[0](x)
        # First Block
        x, lat1 = self.encoder_blocks[1](x)
        # Second Block
        x, lat2 = self.encoder_blocks[2](x)
        # Third Block
        x, lat3 = self.encoder_blocks[3](x)
        # Forth Block
        x, _ = self.encoder_blocks[4](x)
        x = self.encoder_blocks[5](x)
        parameters = self.parameter_encoder(parameters).view(-1, 8, 8, 8, 3).permute(0, 4, 1, 2, 3)

        x = torch.cat((parameters, x), dim=1)

        # First Upscale Block
        up3_skip, gate3 = self.decoder_blocks[0](x, x)
        # After Upsacale Conv
        att3,_ = self.attentionblock3(lat3, gate3)
        up3 = self.decoder_blocks[1](torch.cat((up3_skip, att3),dim=1))
        # Second Upscale Block
        up2_skip, gate2 = self.decoder_blocks[2](up3, up3_skip)
        # After Upscale Conv
        att2,_ = self.attentionblock3(lat2, gate2)
        up2 = self.decoder_blocks[3](torch.cat((up2_skip, att2),dim=1))
        # Third Upscale Block
        up1_skip, gate1 = self.decoder_blocks[4](up2, up2_skip)
        # After Upscale Conv
        att1, attmap = self.attentionblock3(lat1, gate1)
        up1 = self.decoder_blocks[5](torch.cat((up1_skip, att1),dim=1))
        # Last Block
        out, _ = self.decoder_blocks[6](up1, up1_skip)
        # Last Conv
        out = self.decoder_blocks[7](out)

        # DSV Heads
        dsv1 = self.dsv1(gate1)
        dsv2 = self.dsv2(gate2)
        out = self.final(torch.cat((out,dsv1, dsv2),dim=1))
        return out, attmap


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           #nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)

        return self.combine_gates(gate_1), attention_1

class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)
