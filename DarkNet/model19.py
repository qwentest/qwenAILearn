# coding: utf-8 
# @时间   : 2022/1/18 8:41 上午
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


class ConvBNLeakRelu(layers.Layer):
    """构建卷积-BN-Relu6模块，方便网络的构建"""

    def __init__(self, out_channel, kernel_size=3, strides=1, **kwargs):
        super(ConvBNLeakRelu, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=out_channel,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  use_bias=False,
                                  padding='same',
                                  name='conv2d_leakRelu')
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn')
        self.leakRelu = layers.LeakyReLU()

    def call(self, inputs, training=False, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.leakRelu(x)
        return x


class DarkNet19(Model):
    def __init__(self, num_class=4, **kwargs):
        super(DarkNet19, self).__init__(**kwargs)
        self.conv1 = ConvBNLeakRelu(32, 3, 1, input_shape=(224, 224, 3))
        self.maxpool1 = layers.MaxPool2D(2, 2)

        self.conv2 = ConvBNLeakRelu(64, 3)
        self.maxpool2 = layers.MaxPool2D(2, 2)

        self.conv3 = ConvBNLeakRelu(128, 3)
        self.conv4 = ConvBNLeakRelu(64, 1)
        self.conv5 = ConvBNLeakRelu(128, 3)
        self.maxpool3 = layers.MaxPool2D(2, 2)

        self.conv6 = ConvBNLeakRelu(256, 3)
        self.conv7 = ConvBNLeakRelu(128, 1)
        self.conv8 = ConvBNLeakRelu(256, 3)
        self.maxpool4 = layers.MaxPool2D(2, 2)  # 14*14

        self.conv9 = ConvBNLeakRelu(512, 3)
        self.conv10 = ConvBNLeakRelu(256, 1)
        self.conv11 = ConvBNLeakRelu(512, 3)
        self.conv12 = ConvBNLeakRelu(256, 1)
        self.conv13 = ConvBNLeakRelu(256, 3)
        self.maxpool5 = layers.MaxPool2D(2, 2)  # 7 * 7

        self.conv14 = ConvBNLeakRelu(1024, 3)
        self.conv15 = ConvBNLeakRelu(512, 1)
        self.conv16 = ConvBNLeakRelu(1024, 3)
        self.conv17 = ConvBNLeakRelu(512, 1)
        self.conv18 = ConvBNLeakRelu(1024, 3)
        self.conv19 = ConvBNLeakRelu(num_class, 1)

        self.avgpool = layers.GlobalAveragePooling2D()
        # self.flatten = layers.Flatten()
        self.softmax = layers.Softmax()

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.maxpool1(x)

        x = self.conv2(x, training=training)
        x = self.maxpool2(x)

        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.maxpool3(x)

        x = self.conv6(x, training=training)
        x = self.conv7(x, training=training)
        x = self.conv8(x, training=training)
        x = self.maxpool4(x)

        x = self.conv9(x, training=training)
        x = self.conv10(x, training=training)
        x = self.conv11(x, training=training)
        x = self.conv12(x, training=training)
        x = self.conv13(x, training=training)
        x = self.maxpool5(x)

        x = self.conv14(x, training=training)
        x = self.conv15(x, training=training)
        x = self.conv16(x, training=training)
        x = self.conv17(x, training=training)
        x = self.conv18(x, training=training)
        x = self.conv19(x, training=training)

        x = self.avgpool(x)
        # x = self.flatten(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    model = DarkNet19(num_class=4)
    model.build([4, 224, 224, 3])
    print(model.summary())