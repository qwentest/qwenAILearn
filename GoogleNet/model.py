# coding: utf-8 
# @时间   : 2022/1/14 2:05 下午
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


class InceptionV1(layers.Layer):
    """Inception结构类"""

    def __init__(self, conv1x1, conv3x3_reduce, conv3x3, conv5x5_reduce, conv5x5, pool_proj, **kwargs):
        super(InceptionV1, self).__init__(**kwargs)
        self.branch1 = layers.Conv2D(conv1x1, kernel_size=1, strides=1, activation='relu')
        # 先降维再升维
        self.branch2 = Sequential([
            layers.Conv2D(conv3x3_reduce, kernel_size=1, strides=1, activation='relu'),
            layers.Conv2D(conv3x3, kernel_size=3, padding='same', strides=1, activation='relu'),
        ])
        self.branch3 = Sequential([
            layers.Conv2D(conv5x5_reduce, kernel_size=1, strides=1, activation='relu'),
            layers.Conv2D(conv5x5, kernel_size=5, padding='same', strides=1, activation='relu'),
        ])
        self.branch4 = Sequential([
            layers.MaxPool2D(pool_size=3, strides=1, padding='same'),
            layers.Conv2D(pool_proj, kernel_size=1, strides=1, activation='relu'),
        ])

    def call(self, inputs, **kwargs):
        outputs = layers.concatenate(
            [self.branch1(inputs), self.branch2(inputs), self.branch3(inputs), self.branch4(inputs)]
        )
        return outputs


"""
    GoogleNetV1中有辅助节点，本程序没有实现这个分支;
    GoogleNetV1中没有引入BN层，GoogleNetV2中才加入了BN层;
"""


class MyGoogleNetV1(Model):
    def __init__(self, num_class=4):
        super(MyGoogleNetV1, self).__init__()
        # 3个卷积层，9个inception层 * 2，4个最大平均池化，1个平均池化，1个全连接层
        self.layer1 = Sequential(
            [
                layers.Conv2D(64, kernel_size=7, strides=2, activation='relu', padding='same',
                              input_shape=(224, 224, 3), name='conv1'),
                layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='maxpool1'),
                layers.Conv2D(64, kernel_size=1, activation='relu', name='conv2'),
                layers.Conv2D(192, kernel_size=3, padding='same', activation='relu', name='conv3'),
                layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='maxpool2'),
            ]
        )
        self.inception_group2 = Sequential([
            InceptionV1(64, 96, 128, 16, 32, 32, name='inception_3a'),
            InceptionV1(128, 128, 192, 32, 96, 64, name='inception_3b'),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='maxpool3')
        ])

        self.inception_group3 = Sequential([
            InceptionV1(192, 96, 208, 16, 48, 64, name='inception_4a'),
            InceptionV1(160, 112, 224, 24, 64, 64, name='inception_4b'),
            InceptionV1(128, 128, 256, 24, 64, 64, name='inception_4c'),
            InceptionV1(112, 114, 288, 32, 64, 64, name='inception_4d'),
            InceptionV1(256, 160, 320, 32, 128, 128, name='inception_4e'),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='maxpool4')
        ])

        self.inception_group4 = Sequential([
            InceptionV1(256, 160, 320, 32, 128, 128, name='inception_5a'),
            InceptionV1(384, 192, 384, 48, 128, 128, name='inception_5b'),
            layers.AvgPool2D(pool_size=7, strides=1, name='avgpool5')
        ])

        self.flatten = layers.Flatten(name='flatten')
        self.output5 = Sequential([
            layers.Dropout(rate=0.4, name='out_dropout'),
            layers.Dense(num_class, name='out_dense'),
            layers.Softmax()
        ])
    """
    使用5个模块来完成GoogleNet V1的构建
    """
    def call(self, inputs, **kwargs):
        x = self.layer1(inputs)
        x = self.inception_group2(x)
        x = self.inception_group3(x)
        x = self.inception_group4(x)
        x = self.flatten(x)
        x = self.output5(x)
        return x


if __name__ == "__main__":
    model = MyGoogleNetV1()
    model.build([4, 224, 224, 3])
    print(model.summary())
