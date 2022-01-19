# coding: utf-8 
# @时间   : 2022/1/15 2:52 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : model19.py
# @微信   ：qwentest123
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras import Model
import time, json, os


class DWConvBNRelu6(layers.Layer):
    """构建DW卷积-BN-Relu6模块"""

    def __init__(self, pointwise_conv_filters, strides=1, **kwargs):
        super(DWConvBNRelu6, self).__init__(**kwargs)
        self.dw_conv1 = layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides=strides, use_bias=False,
                                               name='dw_conv')
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='dw_conv_bn')
        self.activate1 = layers.ReLU(max_value=6.0, name='dw_conv_relu6')
        # PW卷积，其实就是普通卷积,PW卷积的步长为1
        self.pw_conv2 = layers.Conv2D(filters=pointwise_conv_filters, kernel_size=1, padding='same', use_bias=False,
                                      name='pw_conv')
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='pw_conv_bn')
        self.activate2 = layers.ReLU(max_value=6.0, name='pw_conv_relu6')

    def call(self, inputs, training=False, **kwargs):
        x = self.dw_conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activate1(x)
        x = self.pw_conv2(x)
        x = self.bn2(x)
        x = self.activate2(x)
        return x


class ConvBNRelu6(layers.Layer):
    """构建卷积-BN-Relu6模块，方便网络的构建"""

    def __init__(self, out_channel, kernel_size=3, strides=1, **kwargs):
        super(ConvBNRelu6, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=out_channel,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  use_bias=False,
                                  padding='same',
                                  name='conv2d')
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn')
        self.activate = layers.ReLU(max_value=6.0, name='conv_relu6')

    def call(self, inputs, training=False, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activate(x)
        return x


class MyMobileNetV1(Model):
    def __init__(self, num_class=4, alpha=1, **kwargs):
        super(MyMobileNetV1, self).__init__(**kwargs)
        self.conv1 = ConvBNRelu6(32 * alpha, 3, 2)
        self.dw1 = DWConvBNRelu6(64 * alpha)
        self.dw2 = DWConvBNRelu6(128 * alpha, 2)
        self.dw3 = DWConvBNRelu6(128 * alpha)
        self.dw4 = DWConvBNRelu6(256 * alpha, 2)
        self.dw5 = DWConvBNRelu6(256 * alpha)
        self.dw6 = DWConvBNRelu6(512 * alpha, 2)
        self.dw7 = DWConvBNRelu6(512 * alpha)
        self.dw8_1 = DWConvBNRelu6(512 * alpha)
        self.dw8_2 = DWConvBNRelu6(512 * alpha)
        self.dw8_3 = DWConvBNRelu6(512 * alpha)
        self.dw8_4 = DWConvBNRelu6(512 * alpha)
        self.dw8_5 = DWConvBNRelu6(512 * alpha)
        self.dw9 = DWConvBNRelu6(1024 * alpha, 2)
        # self.pool1 = layers.AvgPool2D(pool_size=7, strides=1)
        self.pool1 = layers.GlobalAvgPool2D(name='pool')## pool + flatten
        self.fc = layers.Dense(num_class, name='fc')
        self.softmax = layers.Softmax()

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.dw3(x)
        x = self.dw4(x)
        x = self.dw5(x)
        x = self.dw6(x)
        x = self.dw7(x)

        x = self.dw8_1(x)
        x = self.dw8_2(x)
        x = self.dw8_3(x)
        x = self.dw8_4(x)
        x = self.dw8_5(x)
        x = self.dw9(x)
        print(x.shape)
        x = self.pool1(x)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        x = self.softmax(x)
        print(x.shape)
        return x


if __name__ == "__main__":
    model = MyMobileNetV1(num_class=4)
    model.build([4, 224, 224, 3])
    print(model.summary())
