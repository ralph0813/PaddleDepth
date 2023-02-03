import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .submodule import convbn_3d, FeatureExtraction, HourGlass, disparity_regression
from .submodule import build_gwc_volume, build_concat_volume


class GwcNet(nn.Layer):
    def __init__(self, max_disp=192, use_concat_volume=True):
        super().__init__()
        self.max_disp = max_disp
        self.use_concat_volume = use_concat_volume
        self.num_groups = 40
        self.concat_channels = 12 if self.use_concat_volume else 0
        self.feature_extraction = FeatureExtraction(
            concat_feature=self.use_concat_volume,
            concat_feature_channel=self.concat_channels
        )
        self.dres0 = nn.Sequential(
            convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
            nn.ReLU(),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.dres1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            convbn_3d(32, 32, 3, 1, 1)
        )

        self.dres2 = HourGlass(32)

        self.dres3 = HourGlass(32)

        self.dres4 = HourGlass(32)

        self.classif0 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3D(32, 1, 3, 1, 1, bias_attr=False)
        )

        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3D(32, 1, 3, 1, 1, bias_attr=False)
        )

        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3D(32, 1, 3, 1, 1, bias_attr=False)
        )

        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3D(32, 1, 3, 1, 1, bias_attr=False)
        )

        for m in self.children():
            if isinstance(m, nn.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
            elif isinstance(m, nn.Conv3D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3D):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        h, w = left.shape[2:]
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.max_disp // 4,
                                      self.num_groups)
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.max_disp // 4)
            volume = paddle.concat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.interpolate(cost0, [self.max_disp, h, w], mode='trilinear', data_format='NCDHW')
            cost0 = paddle.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, axis=1)
            pred0 = disparity_regression(pred0, self.max_disp)

            cost1 = F.interpolate(cost1, [self.max_disp, h, w], mode='trilinear', data_format='NCDHW')
            cost1 = paddle.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, axis=1)
            pred1 = disparity_regression(pred1, self.max_disp)

            cost2 = F.interpolate(cost2, [self.max_disp, h, w], mode='trilinear', data_format='NCDHW')
            cost2 = paddle.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, axis=1)
            pred2 = disparity_regression(pred2, self.max_disp)

            cost3 = F.interpolate(cost3, [self.max_disp, h, w], mode='trilinear', data_format='NCDHW')
            cost3 = paddle.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, axis=1)
            pred3 = disparity_regression(pred3, self.max_disp)
            return [pred0, pred1, pred2, pred3]

        else:
            cost3 = self.classif3(out3)
            cost3 = F.interpolate(cost3, [self.max_disp, h, w], mode='trilinear', data_format='NCDHW')
            cost3 = paddle.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, axis=1)
            pred3 = disparity_regression(pred3, self.max_disp)
            return [pred3]
