"""RGB Inception-I3D model definition for feature extraction.

This module provides the model structure needed to load real Kinetics-pretrained
RGB I3D weights and extract 1024-d clip features. It intentionally does not
download weights or initialize a usable extractor without an explicit checkpoint.

Architecture reference: piergiaj/pytorch-i3d and DeepMind Kinetics-I3D.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, size):
        if size % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        return max(self.kernel_size[dim] - (size % self.stride[dim]), 0)

    def forward(self, x):
        batch, channel, time, height, width = x.size()
        out_time = (time + self.stride[0] - 1) // self.stride[0]
        out_height = (height + self.stride[1] - 1) // self.stride[1]
        out_width = (width + self.stride[2] - 1) // self.stride[2]

        pad_time = self.compute_pad(0, time)
        pad_height = self.compute_pad(1, height)
        pad_width = self.compute_pad(2, width)

        pad_front = pad_time // 2
        pad_back = pad_time - pad_front
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        return F.max_pool3d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding=0,
        activation_fn=F.relu,
        use_batch_norm=True,
        use_bias=False,
        name="unit_3d",
    ):
        super().__init__()
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._padding = padding
        self._stride = stride
        self._kernel_shape = kernel_shape
        self.name = name

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=kernel_shape,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, size):
        if size % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        return max(self._kernel_shape[dim] - (size % self._stride[dim]), 0)

    def forward(self, x):
        batch, channel, time, height, width = x.size()
        pad_time = self.compute_pad(0, time)
        pad_height = self.compute_pad(1, height)
        pad_width = self.compute_pad(2, width)

        pad_front = pad_time // 2
        pad_back = pad_time - pad_front
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super().__init__()

        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=(1, 1, 1),
            name=f"{name}/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=(1, 1, 1),
            name=f"{name}/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=(3, 3, 3),
            name=f"{name}/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=(1, 1, 1),
            name=f"{name}/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=(3, 3, 3),
            name=f"{name}/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=(1, 1, 1),
            name=f"{name}/Branch_3/Conv3d_0b_1x1",
        )
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        num_classes=400,
        spatial_squeeze=True,
        final_endpoint="Logits",
        in_channels=3,
        dropout_keep_prob=0.5,
    ):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint {final_endpoint}")

        super().__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        self.end_points = OrderedDict()
        self.end_points["Conv3d_1a_7x7"] = Unit3D(
            in_channels=in_channels,
            output_channels=64,
            kernel_shape=(7, 7, 7),
            stride=(2, 2, 2),
            name="Conv3d_1a_7x7",
        )
        self.end_points["MaxPool3d_2a_3x3"] = MaxPool3dSamePadding(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=0,
        )
        self.end_points["Conv3d_2b_1x1"] = Unit3D(
            in_channels=64,
            output_channels=64,
            kernel_shape=(1, 1, 1),
            name="Conv3d_2b_1x1",
        )
        self.end_points["Conv3d_2c_3x3"] = Unit3D(
            in_channels=64,
            output_channels=192,
            kernel_shape=(3, 3, 3),
            name="Conv3d_2c_3x3",
        )
        self.end_points["MaxPool3d_3a_3x3"] = MaxPool3dSamePadding(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=0,
        )
        self.end_points["Mixed_3b"] = InceptionModule(
            192,
            [64, 96, 128, 16, 32, 32],
            name="Mixed_3b",
        )
        self.end_points["Mixed_3c"] = InceptionModule(
            256,
            [128, 128, 192, 32, 96, 64],
            name="Mixed_3c",
        )
        self.end_points["MaxPool3d_4a_3x3"] = MaxPool3dSamePadding(
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=0,
        )
        self.end_points["Mixed_4b"] = InceptionModule(
            480,
            [192, 96, 208, 16, 48, 64],
            name="Mixed_4b",
        )
        self.end_points["Mixed_4c"] = InceptionModule(
            512,
            [160, 112, 224, 24, 64, 64],
            name="Mixed_4c",
        )
        self.end_points["Mixed_4d"] = InceptionModule(
            512,
            [128, 128, 256, 24, 64, 64],
            name="Mixed_4d",
        )
        self.end_points["Mixed_4e"] = InceptionModule(
            512,
            [112, 144, 288, 32, 64, 64],
            name="Mixed_4e",
        )
        self.end_points["Mixed_4f"] = InceptionModule(
            528,
            [256, 160, 320, 32, 128, 128],
            name="Mixed_4f",
        )
        self.end_points["MaxPool3d_5a_2x2"] = MaxPool3dSamePadding(
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            padding=0,
        )
        self.end_points["Mixed_5b"] = InceptionModule(
            832,
            [256, 160, 320, 32, 128, 128],
            name="Mixed_5b",
        )
        self.end_points["Mixed_5c"] = InceptionModule(
            832,
            [384, 192, 384, 48, 128, 128],
            name="Mixed_5c",
        )

        for endpoint in self.VALID_ENDPOINTS:
            if endpoint in self.end_points:
                self.add_module(endpoint, self.end_points[endpoint])
            if endpoint == final_endpoint:
                break

        if final_endpoint in ("Logits", "Predictions"):
            self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
            self.dropout = nn.Dropout(dropout_keep_prob)
            self.logits = Unit3D(
                in_channels=1024,
                output_channels=num_classes,
                kernel_shape=(1, 1, 1),
                padding=0,
                activation_fn=None,
                use_batch_norm=False,
                use_bias=True,
                name="Logits",
            )

    def forward(self, x):
        for endpoint in self.VALID_ENDPOINTS:
            if endpoint in self.end_points:
                x = self._modules[endpoint](x)
            if endpoint == self._final_endpoint:
                return x

        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.logits(x)
        if self._spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        if self._final_endpoint == "Logits":
            return x
        return F.softmax(x, dim=1)

    def extract_features(self, x):
        for endpoint in self.VALID_ENDPOINTS:
            if endpoint in self.end_points:
                x = self._modules[endpoint](x)
            if endpoint == "Mixed_5c":
                break
        return F.adaptive_avg_pool3d(x, output_size=(1, 1, 1)).flatten(1)

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(
            in_channels=1024,
            output_channels=num_classes,
            kernel_shape=(1, 1, 1),
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="Logits",
        )
