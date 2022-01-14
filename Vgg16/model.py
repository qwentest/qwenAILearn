# coding: utf-8 
# @时间   : 2022/1/14 8:42 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : model.py
# @微信   ：qwentest123
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras import Model


class MyVgg16(Model):
    def __init__(self, num_class=4):
        super(MyVgg16, self).__init__()
        # 13个卷积积，5个下采样, 3个全连接层
        self.conv1 = Sequential([
            layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', input_shape=(224, 224, 3), activation='relu'),
            layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', input_shape=(224, 224, 3), activation='relu'),
            # 224 * 224 * 64
            layers.MaxPool2D(pool_size=2, strides=2),  # 112 * 112 * 64

            layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),  # 112 * 112 * 128
            layers.MaxPool2D(pool_size=2, strides=2),  # 56 * 56 * 128

            layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'),  # 56 * 56 * 256
            layers.MaxPool2D(pool_size=2, strides=2),  # 28 * 28 * 256

            layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),  # 28 * 28 * 512
            layers.MaxPool2D(pool_size=2, strides=2),  # 14 * 14 * 512

            layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),  # 14 * 14 * 512
            layers.MaxPool2D(pool_size=2, strides=2),  # 7 * 7 * 512
        ])
        self.flatten = layers.Flatten()
        self.FC = Sequential([
            layers.Dense(4096, activation='relu'),
            layers.Dense(4096, activation='relu'),
            layers.Dense(num_class),
            layers.Softmax()
        ])

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.FC(x)
        return x


if __name__ == "__main__":
    model = MyVgg16()
    model.build([4, 224, 224, 3])
    print(model.summary())
