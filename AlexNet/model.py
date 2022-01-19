# coding: utf-8 
# @时间   : 2022/1/13 8:26 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : model19.py
# @微信   ：qwentest123
import tensorflow as tf
import numpy as np
import pandas as pd
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, MaxPool2D
from tensorflow.keras import Model, Sequential, layers


class MyAlextNet(Model):
    def __init__(self, num_classes=2):
        super(MyAlextNet, self).__init__()
        self.layer1 = Sequential(
            [
                layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu', input_shape=(227, 227, 3)),
                layers.MaxPool2D(pool_size=3, strides=2),
                layers.Conv2D(filters=256, kernel_size=5, strides=1, padding="same", activation="relu"),
                layers.MaxPool2D(pool_size=3, strides=2),
                layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
                layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
                layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
                layers.MaxPool2D(pool_size=3, strides=2),
            ]
        )
        self.flatten = layers.Flatten()
        self.classifier = Sequential(
            [
                layers.Dense(4096, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(4096, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(num_classes),
                layers.Softmax()
            ]
        )

    def call(self, inputs, **kwargs):
        x = self.layer1(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = MyAlextNet()
    model.build([4, 227, 227, 3])
    print(model.summary())
