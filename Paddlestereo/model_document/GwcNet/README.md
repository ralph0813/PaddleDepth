# GwcNet(CVPR 2019)

A paddle implementation of the paper Group-wise Correlation Stereo Network, CVPR 19, Xiaoyang Guo, Kai Yang, Wukui Yang,
Xiaogang Wang, and Hongsheng Li [(Paper)](https://ieeexplore.ieee.org/document/8954018/)

## Abstract

Stereo matching estimates the disparity between a rectified image pair, which is of great importance to depth sensing,
autonomous driving, and other related tasks. Previous works built cost volumes with cross-correlation or concatenation
of left and right features across all disparity levels, and then a 2D or 3D convolutional neural network is utilized to
regress the disparity maps. In this paper, we propose to construct the cost volume by group-wise correlation. The left
features and the right features are divided into groups along the channel dimension, and correlation maps are computed
among each group to obtain multiple matching cost proposals, which are then packed into a cost volume. Group-wise
correlation provides efficient representations for measuring feature similarities and will not lose too much information
like full correlation. It also preserves better performance when reducing parameters compared with previous methods. The
3D stacked hourglass network proposed in previous works is improved to boost the performance and decrease the inference
computational cost. Experiment results show that our method outperforms previous methods on Scene Flow, KITTI 2012, and
KITTI 2015 datasets.

### Train

1. Pre-training

```shell
$ ./Scripts/start_train_sceneflow_stereo_net_multi.sh
```

2. Fine-tuning (KITTI 2012)

```shell
$ ./Scripts/start_train_kitti2012_stereo_net_multi.sh
```

### Test

1. KITTI2012

```shell
$ ./Scripts/start_test_kitti2012_stereo_net.sh
```

**Note**: Plase update .csv file in `Datasets/Stereo` and choose `--modelName PCWNet`

## Models

Models will be open-sourced later

## Link

we also provide the official pytorch implementation in this [website](https://github.com/xy-guo/GwcNet)

## Citation

If you find this code useful in your research, please cite:

```
@inproceedings{guo2019group,
  title={Group-wise Correlation Stereo Network},
  author={Guo, Xiaoyang and Yang, Kai and Yang, Wukui and Wang, Xiaogang and Li, Hongsheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3273--3282},
  year={2019}
}
```
