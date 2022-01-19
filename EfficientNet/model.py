# coding: utf-8 
# @时间   : 2022/1/18 2:09 下午
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
import time, json, os
from typing import Union


def correct_pad(input_size: Union[int, tuple], kernel_size: int):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    Arguments:
      input_size: Input tensor size.
      kernel_size: An integer or tuple/list of 2 integers.
    Returns:
      A tuple.
    """

    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    kernel_size = (kernel_size, kernel_size)

    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

def MBConv(inputs, activation: str = 'swish',
           name: str = "", input_channel: int = 32,
           output_channel: int = 16, kernel_size: int = 3,
           strides: int = 1, expand_ratio: int = 1, use_se: bool = True,
           se_ration: float = 1 / 4.,
           drop_rate=0.
           ):
    # 第1个卷积核是输入特征矩阵channel的n的倍数.
    filters = input_channel * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', use_bias=False,
                          name=name + "/expand_conv")(inputs)
    else:
        x = inputs
    if strides == 2:
        # 对图像进行补0对齐
        x = layers.ZeroPadding2D(padding=correct_pad(filters, kernel_size),
                                 name=name + "dwconv_pad")(x)
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                               padding='same' if strides == 1 else 'valid',
                               use_bias=False,
                               name=name + "dwconv"
                               )(x)
    x = layers.BatchNormalization(name=name + "bn")(x)
    x = layers.Activation(activation, name=name + "activation")(x)

    if use_se:
        filters_se = int(input_channel * se_ration)
        se = layers.GlobalAveragePooling2D(name=name + "se")(x)
        se = layers.Reshape((1, 1, filters), name=name + "se_reshape")(se)
        # 第一个fc使用的是swish激活函数
        se = layers.Conv2D(filters=filters_se,
                           kernel_size=1,
                           padding="same",
                           activation=activation,
                           name=name + "se_reduce")(se)
        # 第二个fc使用的是sigmoid函数
        se = layers.Conv2D(filters=filters,
                           kernel_size=1,
                           padding="same",
                           activation="sigmoid",
                           name=name + "se_expand")(se)
        x = layers.multiply([x, se], name=name + "se_excite")
    # 再经过一个1x1的卷积
    x = layers.Conv2D(filters=output_channel,
                      kernel_size=1,
                      padding="same",
                      use_bias=False,
                      name=name + "project_conv")(x)
    x = layers.BatchNormalization(name=name + "project_bn")(x)
    # 只有s=1,并且输入通道与输出通道数相同时，才使用残差结构
    # 并且只有在shortcut时才使用dropout
    if strides == 1 and input_channel == output_channel:
        if drop_rate > 0:
            x = layers.Dropout(rate=drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + "add")
    # MBconv V2之后，直接就是线性输出，不再经过relu函数
    return x


import math

def efficient_net(width_coefficient, depth_coefficient, input_shape=(224, 224, 3),
                  dropout_rate=0.2, drop_connect_rate=0.2, activation="swish",
                  model_name="efficientnet", num_class=4
                  ):
    # kernel_size, repeats, in_channel, out_channel, exp_ratio, strides, SE
    block_args = [[3, 1, 32, 16, 1, 1, False],
                  [3, 2, 16, 24, 6, 2, False],
                  [5, 2, 24, 40, 6, 2, False],
                  [3, 3, 40, 80, 6, 2, True],
                  [5, 3, 80, 112, 6, 1, True],
                  [5, 4, 112, 192, 6, 2, True],
                  [3, 1, 192, 320, 6, 1, True]]

    # 以下两个函数是用来确定，宽度、深度系数之后，能够得到一个整数
    def round_filters(filters, divisor=8):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    image_input = layers.Input(shape=input_shape)
    # 第一个卷积
    x = layers.ZeroPadding2D(padding=correct_pad(input_shape[:2], 3),
                             name="stem_conv_pad")(image_input)
    x = layers.Conv2D(filters=round_filters(32),
                      kernel_size=3,
                      strides=2,
                      padding="valid",
                      use_bias=False,
                      name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation(activation, name="stem_activation")(x)
    # 开始构建block
    b = 0
    num_blocks = float(sum(round_repeats(i[1]) for i in block_args))  # 7
    for i, args in enumerate(block_args):
        assert args[1] > 0
        # 需要保证输入的通道数是8的倍数
        args[2] = round_filters(args[2])  # input_channel
        args[3] = round_filters(args[3])  # output_channel
        for j in range(round_repeats(args[1])):
            # 重复的次数
            # drop_connect_rate * b / num_blocks，实现线性dropout
            x = MBConv(x, activation=activation, drop_rate=drop_connect_rate * b / num_blocks,
                       name='block{}{}'.format(i + 1, j),
                       kernel_size=args[0],
                       input_channel=args[2] if j == 0 else args[3],  # 第1次重复时，使用args[2]的通道数，再次重复时使用输出通道数
                       output_channel=args[3],
                       expand_ratio=args[4],
                       strides=args[5] if j == 0 else 1,
                       use_se=args[6]
                       )

            b += 1
    # last
    x = layers.Conv2D(filters=1280,
                      kernel_size=1,
                      strides=1,
                      padding="same",
                      use_bias=False,
                      name="last_conv")(x)
    x = layers.BatchNormalization(name='lastBn')(x)
    x = layers.Activation(activation, name='last_activation')(x)

    x = layers.GlobalAveragePooling2D(name='global_avg_pool_last')(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="top_dropout")(x)
    x = layers.Dense(units=num_class,
                     activation="softmax",
                     name="predictions")(x)

    model = Model(inputs=image_input, outputs=x, name=model_name)
    return model


def efficientnet_b0(num_class=4,
                    input_shape=(224, 224, 3)):
    # https://storage.googleapis.com/keras-applications/efficientnetb0.h5
    return efficient_net(width_coefficient=1.0,
                         depth_coefficient=1.0,
                         input_shape=input_shape,
                         dropout_rate=0.2,
                         model_name="efficientnetb0",
                         num_class=num_class)


if __name__ == "__main__":
    model = efficientnet_b0(num_class=4)
    model.build([4, 224, 224, 3])
    print(model.summary())
