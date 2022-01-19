# coding: utf-8 
# @时间   : 2022/1/19 2:00 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : modelcsp.py
# @微信   ：qwentest123
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras import Model
import time, json, os


class Mish(layers.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        x = inputs * tf.math.tanh(tf.math.softplus(inputs))
        return x


class ConvBNMish(layers.Layer):
    def __init__(self, out_channel, kernel_size, strides, padding='same', **kwargs):
        super(ConvBNMish, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=out_channel,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  use_bias=False,
                                  padding=padding,
                                  name='conv2d')
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn')
        self.activate = Mish()

    def call(self, inputs, training=False, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activate(x)
        return x


class ResBlock(layers.Layer):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.block = Sequential([
            ConvBNMish(in_channel, kernel_size=1, strides=1, padding='valid'),
            ConvBNMish(out_channel, kernel_size=3, strides=1, padding='same')
        ])

    def call(self, inputs, *args, **kwargs):
        return inputs + self.block(inputs)


class CSPResBlock(layers.Layer):
    def __init__(self, in_channel, out_channel, num_block, first=False, **kwargs):
        super(CSPResBlock, self).__init__(**kwargs)
        # csp中第一个layer1是用来做下采样的。其s=2
        self.downSample = ConvBNMish(in_channel, kernel_size=3, strides=2)  # 32
        # CSPRes1中输入的是64，残差之后输出的仍然是64。
        # CSPRes2 128->64
        # CSPRes3 256->128
        # CSPRes4 512->256
        # CSPRes5 1024->512
        if first:
            self.split_conv0 = ConvBNMish(out_channel, kernel_size=1, strides=1)  # 64
            self.split_conv1 = ConvBNMish(out_channel, kernel_size=1, strides=1)
            self.bocks_conv = Sequential([
                ResBlock(in_channel, out_channel),
                ConvBNMish(out_channel, kernel_size=1, strides=1)
            ])
            self.cat_conv = ConvBNMish(out_channel, kernel_size=1, strides=1)
        else:
            # 从第二个csp开始，输出维度是输入维度的一半
            self.split_conv0 = ConvBNMish(out_channel // 2, kernel_size=1, strides=1)  # 128 // 2
            self.split_conv1 = ConvBNMish(out_channel // 2, kernel_size=1, strides=1)
            self.bocks_conv = Sequential([
                *[ResBlock(out_channel // 2, out_channel // 2) for _ in range(num_block)],
                ConvBNMish(out_channel // 2, kernel_size=1, strides=1)
            ])
            self.cat_conv = ConvBNMish(out_channel, kernel_size=1, strides=1)

    def call(self, inputs, *args, **kwargs):
        # layer_1
        x = self.downSample(inputs)
        # 两个分支
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        # 残差
        x0 = self.bocks_conv(x0)
        out = layers.Concatenate()([x1, x0])
        # 再进过一个1x1
        out = self.cat_conv(out)
        return out


class CSPDarkNet53(Model):
    def __init__(self, layer_num=[1, 2, 8, 8, 4], num_class=4):
        """
        :param layer_num: csp重复出现的次数
        :param num_class: 分类数
        :param kwargs:
        """
        super(CSPDarkNet53, self).__init__()
        self.conv1 = ConvBNMish(32, kernel_size=3, strides=1)
        # 输入csp的输入模块中
        self.stage = [
            CSPResBlock(64, 64, layer_num[0], first=True),
            CSPResBlock(128, 128, layer_num[1]),
            CSPResBlock(256, 256, layer_num[2]),
            CSPResBlock(512, 512, layer_num[3]),
            CSPResBlock(1024, 1024, layer_num[4])
        ]
        self.global_pool = layers.GlobalAvgPool2D()
        self.last_conv = ConvBNMish(num_class, kernel_size=1, strides=1)
        self.flatten = layers.Flatten()
        self.softmax = layers.Softmax()

    def call(self, inputs, training=False, num_class=4):
        x = self.conv1(inputs)  # 608x608x32
        x = self.stage[0](x)  # 304x304x64
        x = self.stage[1](x)  # 152x152x128
        d1 = self.stage[2](x)  # 76x76x256  ****
        d2 = self.stage[3](d1)  # 38x38x512  ****
        d3 = self.stage[4](d2)  # 19x19x1024 ****
        x = self.global_pool(d3)
        x = layers.Reshape((1, 1, 1024))(x)
        x = self.last_conv(x)
        x = self.flatten(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = CSPDarkNet53(num_class=4)
    model.build([4, 224, 224, 3])
    print(model.summary())
