import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding=dilation if dilation > 1 else pad, dilation=dilation, bias_attr=False),
        nn.BatchNorm2D(out_channels)
    )


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, bias_attr=False),
        nn.BatchNorm3D(out_channels)
    )


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super().__init__()
        self.conv1 = nn.Sequential(
            convbn(inplanes, planes, 3, stride, pad, dilation),
            nn.ReLU()
        )

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class FeatureExtraction(nn.Layer):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super().__init__()
        self.concat_feature = concat_feature
        self.inplanes = 32

        self.firstconv = nn.Sequential(
            convbn(3, 32, 3, 2, 1, 1),
            nn.ReLU(),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU()
        )
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(),
                                          nn.Conv2D(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias_attr=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = paddle.concat((l2, l3, l4), axis=1)

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}


class HourGlass(nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv3DTranspose(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2,
                               bias_attr=False),
            nn.BatchNorm3D(in_channels * 2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv3DTranspose(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias_attr=False),
            nn.BatchNorm3D(in_channels)
        )

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2))
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x))
        return conv6


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).reshape([B, num_groups, channels_per_group, H, W]).mean(axis=2)
    assert cost.shape == [B, num_groups, H, W], f"cost shape is wrong {cost.shape}, should be {(B, num_groups, H, W)}"
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, max_disp, num_groups):
    B, C, H, W = refimg_fea.shape
    # paddle.zeros([B, num_groups, max_disp, H, W], dtype=refimg_fea.dtype)
    volume = paddle.zeros([B, num_groups, max_disp, H, W], dtype=refimg_fea.dtype)
    for i in range(max_disp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(
                refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                num_groups
            )
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    return volume


def build_concat_volume(refimg_fea, targetimg_fea, max_disp):
    B, C, H, W = refimg_fea.shape
    volume = paddle.zeros([B, 2 * C, max_disp, H, W], dtype=refimg_fea.dtype)
    for i in range(max_disp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    return volume


def disparity_regression(x, max_disp):
    assert len(x.shape) == 4
    disp_values = paddle.arange(0, max_disp, dtype=x.dtype)
    disp_values = disp_values.reshape([1, max_disp, 1, 1])
    return paddle.sum(x * disp_values, 1, keepdim=False)
