import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class ConvDropoutNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), is_dropout=False):
        super(ConvDropoutNormReLU, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=tuple([(i - 1) // 2 for i in kernel_size]))
        self.norm = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if is_dropout:
            self.dropout = nn.Dropout3d(p=0.2, inplace=True)
        else:
            self.dropout = Identity()
        self.all = nn.Sequential(self.conv, self.dropout, self.norm, self.nonlin)

    def forward(self, x):
        return self.all(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 stride1=(1, 1, 1), stride2=(1, 1, 1), is_dropout=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride1,
                               padding=tuple([(i - 1) // 2 for i in kernel_size]))
        self.norm1 = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if is_dropout:
            self.dropout = nn.Dropout3d(p=0.2, inplace=True)
        else:
            self.dropout = Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride2,
                               padding=tuple([(i - 1) // 2 for i in kernel_size]))
        self.norm2 = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        if any(i != 1 for i in stride1) or in_channels!=out_channels:
            self.downsample_skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride1, bias=False),
                nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            )
        else:
            self.downsample_skip = lambda x: x

    def forward(self, x):
        residual = x
        out = self.dropout(self.conv1(x))
        out = self.nonlin1(self.norm1(out))
        out = self.norm2(self.conv2(out))
        residual = self.downsample_skip(residual)
        out += residual
        return self.nonlin2(out)


class EncoderBlock(nn.Module):
    def __init__(self, stage, in_channels, out_channels, kernel_size=(3, 3, 3),
                 conv1_stride=(2, 2, 2), conv2_stride=(1, 1, 1), is_dropout=False):
        super(EncoderBlock, self).__init__()
        ops = []
        if stage == 0:
            ops.append(ConvDropoutNormReLU(in_channels, out_channels, kernel_size, conv1_stride, is_dropout))
            ops.append(ResidualBlock(out_channels, out_channels, kernel_size, conv2_stride, conv2_stride, is_dropout))
        elif stage > 0:
            ops.append(ResidualBlock(in_channels, out_channels, kernel_size, conv1_stride, conv2_stride, is_dropout))
            # Only one ResidualBlock at one stage
            for _ in range(1):
                ops.append(ResidualBlock(out_channels, out_channels, kernel_size, conv2_stride, conv2_stride, is_dropout))
            # If your GPU memory is large, you can do like the following setting
            # Increase the number of ResidualBlock based on the stage
            # for _ in range(stage):
            #     ops.append(ResidualBlock(out_channels, out_channels, kernel_size, conv2_stride, conv2_stride, is_dropout))
        self.StackedConvLayers = nn.Sequential(*ops)

    def forward(self, x):
        return self.StackedConvLayers(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)):
        super(Upsample, self).__init__()
        self.ConvTrans = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

    def forward(self, x):
        return self.norm(self.ConvTrans(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 conv_stride=(1, 1, 1), is_dropout=False):
        super(DecoderBlock, self).__init__()
        self.conv = ConvDropoutNormReLU(in_channels, out_channels, kernel_size, conv_stride, is_dropout)

    def forward(self, x):
        return self.conv(x)


class JPAmodule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(JPAmodule, self).__init__()
        self.CA1 = ChannelAttention(channel, reduction)
        self.CA2 = ChannelAttention(channel, reduction)
        self.CA3 = ChannelAttention(channel * 2, reduction)
        self.gate_x1 = nn.Conv3d(channel * 2, 1, kernel_size=1, bias=True)
        self.gate_x2 = nn.Conv3d(channel * 2, 1, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.mid = ResidualBlock(channel, channel, kernel_size=(1, 1, 1))
        self.out = ResidualBlock(channel, channel, kernel_size=(1, 1, 1))

    def forward(self, x1, x2):
        x1_ca = self.CA1(x1)
        x2_ca = self.CA2(x2)
        x1_x2_sub = torch.sub(x1_ca, x2_ca)

        cat_fea = torch.cat([x1, x2], dim=1)
        cat_fea = self.CA3(cat_fea)
        x1_weight = self.gate_x1(cat_fea)
        x2_weight = self.gate_x2(cat_fea)
        weights = self.softmax(torch.cat((x1_weight, x2_weight), dim=1))
        x1_weight, x2_weight = weights[:, 0:1, :, :, :], weights[:, 1:2, :, :, :]
        x1_x2_add = x1.mul(x1_weight) + x2.mul(x2_weight)

        x1_x2_ag = torch.add(x1_x2_add, x1_x2_sub)
        y = self.out(self.mid(x1_x2_ag))
        return y


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        out = x * y.expand(x.size())
        return out


class JPAnet(nn.Module):
    def __init__(self, in_channels, out_channels, is_dropput=False):
        # only two DCE images(C1, C0) as inputs (without BREAST PRED)
        super(JPAnet, self).__init__()
        self.encoder1 = EncoderBlock(stage=0, in_channels=in_channels, out_channels=32, conv1_stride=(1, 1, 1), is_dropout=is_dropput)
        self.encoder2 = EncoderBlock(stage=1, in_channels=32, out_channels=2 * 32, is_dropout=is_dropput)
        self.encoder3 = EncoderBlock(stage=2, in_channels=2 * 32, out_channels=4 * 32, is_dropout=is_dropput)
        self.encoder4 = EncoderBlock(stage=3, in_channels=4 * 32, out_channels=8 * 32, is_dropout=is_dropput)
        self.encoder5 = EncoderBlock(stage=4, in_channels=8 * 32, out_channels=320, is_dropout=is_dropput)

        self.attention1 = JPAmodule(32)
        self.attention2 = JPAmodule(2 * 32)
        self.attention3 = JPAmodule(4 * 32)
        self.attention4 = JPAmodule(8 * 32)
        self.attention5 = JPAmodule(320)

        self.up1 = Upsample(320, 8 * 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up2 = Upsample(8 * 32, 4 * 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up3 = Upsample(4 * 32, 2 * 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up4 = Upsample(2 * 32, 1 * 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.decoder1 = DecoderBlock(512, 256, is_dropout=is_dropput)
        self.decoder2 = DecoderBlock(256, 128, is_dropout=is_dropput)
        self.decoder3 = DecoderBlock(128, 64, is_dropout=is_dropput)
        self.decoder4 = DecoderBlock(64, 32, is_dropout=is_dropput)

        self.out1 = nn.Conv3d(256, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.out2 = nn.Conv3d(128, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.out3 = nn.Conv3d(64, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.out4 = nn.Conv3d(32, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)


    def forward(self, input):
        channel_input_1 = input[:, 0:1]
        x1_1 = self.encoder1(channel_input_1)
        x2_1 = self.encoder2(x1_1)
        x3_1 = self.encoder3(x2_1)
        x4_1 = self.encoder4(x3_1)
        x5_1 = self.encoder5(x4_1)

        channel_input_2 = input[:, 1:2]
        x1_2 = self.encoder1(channel_input_2)
        x2_2 = self.encoder2(x1_2)
        x3_2 = self.encoder3(x2_2)
        x4_2 = self.encoder4(x3_2)
        x5_2 = self.encoder5(x4_2)

        x1 = self.attention1(x1_1, x1_2)
        x2 = self.attention2(x2_1, x2_2)
        x3 = self.attention3(x3_1, x3_2)
        x4 = self.attention4(x4_1, x4_2)
        x5 = self.attention5(x5_1, x5_2)

        x5_up = self.up1(x5)
        x4_x5 = torch.cat((x4, x5_up), dim=1)
        y1 = self.decoder1(x4_x5)

        y1_up = self.up2(y1)
        x3_y1 = torch.cat((x3, y1_up), dim=1)
        y2 = self.decoder2(x3_y1)

        y2_up = self.up3(y2)
        x2_y2 = torch.cat((x2, y2_up), dim=1)
        y3 = self.decoder3(x2_y2)

        y3_up = self.up4(y3)
        x1_y3 = torch.cat((x1, y3_up), dim=1)
        y4 = self.decoder4(x1_y3)

        out1 = self.out1(y1)
        out2 = self.out2(y2)
        out3 = self.out3(y3)
        out4 = self.out4(y4)

        out_list = [out4, out3, out2, out1]
        return out_list


if __name__ == '__main__':
    import os
    # import torchsummary
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = JPAnet(1, 2, is_dropput=False).cuda()
    x = torch.ones((1, 2, 96, 96, 96)).cuda()
    y = model(x)
    print(1)
    # torchsummary.summary(model, (2, 288, 96, 80))
    # torchsummary.summary(model, (2, 96, 96, 96))



