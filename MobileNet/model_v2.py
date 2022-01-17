# coding: utf-8 
# @时间   : 2022/1/16 2:29 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : model_v2.py
# @微信   ：qwentest123
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras import Model
import time, json, os

from model import ConvBNRelu6


class InvertedResidual(layers.Layer):
    """
    倒残差结构。因为mobileNet主要使用DW卷积，如果先降维，则特征相对来说较少。
    Relu函数会将负值去除，则损失了一部分特征信息，所以进行线性输出。
    """

    def __init__(self, in_channel, out_channel, strides, expand_ratio, **kwargs):
        super(InvertedResidual, self).__init__()
        # 利用扩展因子
        self.true_channel = in_channel * expand_ratio
        # 只有在步长为1，且输出与输出channel一直时才使用shorCut
        self.is_shortcut = strides == 1 and in_channel == out_channel
        layer_list = []
        if expand_ratio != 1:
            layer_list.append(ConvBNRelu6(out_channel=self.true_channel, kernel_size=1, name='expand'))
        # DW卷积之后，1*1的卷积进行线性输出，不再经过relu函数
        layer_list.extend([
            layers.DepthwiseConv2D(kernel_size=3, padding='same', strides=strides, use_bias=False, name='dw_conv'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='dw/bn1'),
            layers.ReLU(max_value=6),
            layers.Conv2D(filters=out_channel, kernel_size=1, strides=1, padding='same', use_bias=False,
                          name='conv_line'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='dw/bn2')
        ])

        self.branch = Sequential(layer_list, name='branch_conv')

    def call(self, inputs, training=False, **kwargs):
        # 只有步长为1，才能进行shortcut
        if self.is_shortcut:
            # 将输出与InvertedResidual进行残差输出
            return inputs + self.branch(inputs, training=training)
        else:
            return self.branch(inputs, training=training)


def _make_divisible(channel, divisor=8, min_ch=None):
    """
    因为mobile中有扩展因子t，通过个方法保证扩展因子能够被8整除.
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(channel + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * channel:
        new_ch += divisor
    return new_ch


def MyMobileNetV2(im_height=224,
                  im_width=224,
                  num_class=4,
                  alpha=1.0,
                  round_nearest=8,
                  include_top=True):
    block = InvertedResidual
    input_channel = _make_divisible(32 * alpha, round_nearest)
    last_channel = _make_divisible(1280 * alpha, round_nearest)
    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    #
    x = ConvBNRelu6(input_channel, strides=2, name='Conv')(input_image)
    #
    for idx, (t, c, n, s) in enumerate(inverted_residual_setting):
        output_channel = _make_divisible(c * alpha, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            # 输入的通道数是输出的通道数
            x = block(x.shape[-1],
                      output_channel,
                      stride,
                      expand_ratio=t)(x)
    x = ConvBNRelu6(last_channel, kernel_size=1, name='Conv_1')(x)

    if include_top is True:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(num_class)(x)
        output = layers.Softmax()(x)
    else:
        output = x

    model = Model(inputs=input_image, outputs=output)
    return model


if __name__ == "__main__":
    model = MyMobileNetV2(num_class=4)
    model.build([4, 224, 224, 3])
    print(model.summary())
