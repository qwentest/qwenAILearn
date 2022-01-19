# coding: utf-8 
# @时间   : 2022/1/18 9:57 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : model53.py
# @微信   ：qwentest123
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras import Model
import time, json, os


class ConvBNLeakRelu(layers.Layer):
    """构建卷积-BN-Relu6模块，方便网络的构建"""

    def __init__(self, out_channel, kernel_size=3, strides=1, padding='same', **kwargs):
        super(ConvBNLeakRelu, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=out_channel,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  use_bias=False,
                                  padding=padding,
                                  name='conv2d_leakRelu')
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn')
        self.leakRelu = layers.LeakyReLU()

    def call(self, inputs, training=False, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.leakRelu(x)
        return x


# 先构建残差结构

def ResidualBlock(x, input_channel, output_channels, counts):
    oldx = ConvBNLeakRelu(output_channels, 3, strides=2, padding='same')(x)
    for y in range(counts):
        x = ConvBNLeakRelu(input_channel, kernel_size=1, strides=1, padding='valid')(oldx)
        x = ConvBNLeakRelu(output_channels, kernel_size=3, strides=1, padding='same')(x)
        x = layers.Add()([oldx, x])
    return x


def DarkNet53(im_height=416, im_width=416, num_class=4):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    # 先是1个普通卷积
    x = ConvBNLeakRelu(32, 3, 1)(input_image)
    # 第一个残差结构
    x = ResidualBlock(x, 32, 64, 1)  # 208 * 208 * 64
    # 第二个残差结构重复2次
    x = ResidualBlock(x, x.shape[-1], 128, 2)  # 104 * 104 * 128
    # 第三个残差结构重复8次
    x = ResidualBlock(x, x.shape[-1], 256, 8)  # 52 * 52 * 256
    # 第四个残差结构重复8次
    x = ResidualBlock(x, x.shape[-1], 512, 8)  # 26 * 26 * 512
    # 第五个残差结构重复4次
    x = ResidualBlock(x, x.shape[-1], 1024, 4)  # 13 * 13 * 1024

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 1024))(x)
    x = ConvBNLeakRelu(num_class, kernel_size=1, strides=1, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Softmax()(x)
    model = Model(inputs=input_image, outputs=x)
    return model


if __name__ == "__main__":
    model = DarkNet53(num_class=4,im_width=256,im_height=256)
    model.build([4, 256, 256, 3])
    print(model.summary())
