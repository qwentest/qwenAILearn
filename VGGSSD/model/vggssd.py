# coding: utf-8 
# @时间   : 2022/1/20 1:21 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : vggssd.py
# @微信   ：qwentest123
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras import Model
import time, json, os

"""
1.根据VGG16进行了更改，将FC层变成了卷积层。
2.FC6换成的是一个空洞卷积，主要目标是为了扩大感受野
"""


def VGGSSD(inputs):
    net = {}
    net['input'] = inputs
    # block1 300 * 300 * 3 -> 150,150,64
    net['conv1_1'] = layers.Conv2D(filters=64, kernel_size=3,
                                   strides=1, padding='same', activation='relu')(net['input'])
    net['conv1_2'] = layers.Conv2D(filters=64, kernel_size=3,
                                   strides=1, padding='same', activation='relu')(net['conv1_1'])
    net['pool1'] = layers.MaxPool2D(pool_size=2, strides=2,padding='same')(net['conv1_2'])

    # block2 150,150,64 -> 75,75,128
    net['conv2_1'] = layers.Conv2D(filters=128, kernel_size=3,
                                   strides=1, padding='same', activation='relu')(net['pool1'])
    net['conv2_2'] = layers.Conv2D(filters=128, kernel_size=3,
                                   strides=1, padding='same', activation='relu')(net['conv2_1'])
    net['pool2'] = layers.MaxPool2D(pool_size=2, strides=2,padding='same')(net['conv2_2'])

    # block3 75,75,128->38,38,256
    net['conv3_1'] = layers.Conv2D(filters=256, kernel_size=3,
                                   strides=1, padding='same', activation='relu')(net['pool2'])
    net['conv3_2'] = layers.Conv2D(filters=256, kernel_size=3,
                                   strides=1, padding='same', activation='relu')(net['conv3_1'])
    net['conv3_3'] = layers.Conv2D(filters=256, kernel_size=3,
                                   strides=1, padding='same', activation='relu')(net['conv3_2'])
    net['pool3'] = layers.MaxPool2D(pool_size=2, strides=2,padding='same')(net['conv3_3'])

    # block4 38,38,256 -> 19,19,512,,这一层的输出会拿来做预测
    net['conv4_1'] = layers.Conv2D(filters=512, kernel_size=3,
                                   strides=1, padding='same', activation='relu')(net['pool3'])
    net['conv4_2'] = layers.Conv2D(filters=512, kernel_size=3,
                                   strides=1, padding='same', activation='relu')(net['conv4_1'])
    # 38 * 38 * 512,上面的maxpool需要为same，不然这里会是 37*37*512
    net['conv4_3'] = layers.Conv2D(filters=512, kernel_size=3,
                                   strides=1, padding='same', activation='relu', name='conv4_3')(net['conv4_2'])  # 这一层会拿来做小目标的预测
    net['pool4'] = layers.MaxPool2D(pool_size=2, strides=2,padding='same')(net['conv4_3'])

    # block5
    net['conv5_1'] = layers.Conv2D(filters=512, kernel_size=3, strides=1,
                                   padding='same', activation='relu')(net['pool4'])
    net['conv5_2'] = layers.Conv2D(filters=512, kernel_size=3, strides=1,
                                   padding='same', activation='relu')(net['conv5_1'])
    net['conv5_3'] = layers.Conv2D(filters=512, kernel_size=3, strides=1,
                                   padding='same', activation='relu')(net['conv5_2'])
    # 这里修改为池化为3，步长为1,same
    net['pool5'] = layers.MaxPool2D(pool_size=3, strides=1,padding='same')(net['conv5_3'])

    # FC6，使用的是空洞卷积，扩大感受野
    net['fc6'] = layers.Conv2D(filters=1024, kernel_size=3, strides=1,
                               padding='same', activation='relu', dilation_rate=(6, 6),name='fc6')(net['pool5'])

    # FC7 是用的FC7，不是FC6  19 * 19 * 1024用来做预测
    net['fc7'] = layers.Conv2D(filters=1024, kernel_size=1, strides=1,
                               padding='same', activation='relu', name='fc7')(net['fc6'])

    # FC8
    net['conv8_1'] = layers.Conv2D(filters=256, kernel_size=1, strides=1,
                                   padding='same', activation='relu', name='conv8_1')(net['fc7'])
    # Conv8_2，10 * 10 * 512用来做预测
    net['conv8_2'] = layers.Conv2D(filters=512, kernel_size=3, strides=2,
                                   padding='same', activation='relu', name='conv8_2')(net['conv8_1'])

    # conv9
    net['conv9_1'] = layers.Conv2D(filters=128, kernel_size=1, strides=1,
                                   padding='same', activation='relu', name='conv9_1')(net['conv8_2'])
    # Conv8_2，5 * 5 * 256用来做预测
    net['conv9_2'] = layers.Conv2D(filters=256, kernel_size=3, strides=2,
                                   padding='same', activation='relu', name='conv9_2')(net['conv9_1'])

    # conv10
    net['conv10_1'] = layers.Conv2D(filters=128, kernel_size=1, strides=1,
                                    padding='same', activation='relu', name='conv10_1')(net['conv9_2'])
    # Conv10_2，3 * 3 * 256用来做预测，注意这里要padding='valid'才会变成3x3，所以这里有信息的丢失
    net['conv10_2'] = layers.Conv2D(filters=256, kernel_size=3, strides=1,
                                    padding='valid', activation='relu', name='conv10_2')(net['conv10_1'])

    # conv11
    net['conv11_1'] = layers.Conv2D(filters=128, kernel_size=1, strides=1,
                                    padding='same', activation='relu', name='conv11_1')(net['conv10_2'])
    # Conv11_2，1 * 1 * 256用来做预测，注意这里要padding='valid'才会变成3x3，所以这里有信息的丢失
    net['conv11_2'] = layers.Conv2D(filters=256, kernel_size=3, strides=1,
                                    padding='valid', activation='relu', name='conv11_2',)(net['conv11_1'])

    return net

if __name__ == "__main__":
    img_input = layers.Input(shape=(300,300,3))
    x = VGGSSD(img_input)
    model = Model(img_input, x, name="ShuffleNetV2")
    model.build([4, 300, 300, 3])
    print(model.summary())