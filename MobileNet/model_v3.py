# coding: utf-8 
# @时间   : 2022/1/17 9:43 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : model_v3.py
# @微信   ：qwentest123
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras import Model
import time, json, os

from typing import Union  # 返回一个集合
from functools import partial

"""
from functools import partial
函数的参数合并到新的亦是中。
参考：https://www.jianshu.com/p/63f09d1221a8

def add(x, y, z):
    print x, y, z
    return x + y + z

add1 = partial(add, 10)
print(add1(20, 30))

输出：
10 20 30
60
"""

from model_v2 import _make_divisible

"""
2.0中没有实现h-swish激活函数，需要自己实现一下
"""


class HardSigmoid(layers.Layer):
    def __init__(self, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.relu6 = layers.ReLU(max_value=6.)

    def call(self, inputs, *args, **kwargs):
        x = self.relu6(inputs + 3) * (1. / 6)
        return x


class HardSwich(layers.Layer):
    def __init__(self, **kwargs):
        super(HardSwich, self).__init__(**kwargs)
        self.hard_sigmoid = HardSigmoid()

    def call(self, inputs, **kwargs):
        x = self.hard_sigmoid(inputs) * inputs
        return x


"""
加入了SE模块:
1.先进行一个全局平均池化
2.第一个全连接使用relu激活
2.第二个全连接使用hard-sigmoid激活函数
"""


def _se_block(inputs, filters, prefix, se_ratio=1 / 4.):
    x = layers.GlobalAveragePooling2D(name=prefix + 'se/avgpool')(inputs)  # pool+flatten
    # [batch,channel]->[batch,1,1,channel]
    x = layers.Reshape((1, 1, filters))(x)
    # fc1，第一个全连接取的是filters的1/4
    x = layers.Conv2D(filters=_make_divisible(filters * se_ratio),
                      kernel_size=1,
                      padding='same',
                      name=prefix + 'se/conv')(x)
    x = layers.ReLU(name=prefix + 'se/relu')(x)

    # fc2 ，第二个filters是所有
    x = layers.Conv2D(filters=filters,
                      kernel_size=1,
                      padding='same',
                      name=prefix + 'se/conv2')(x)
    x = HardSigmoid(name=prefix + 'se/hardsigmoid2')(x)
    # tf.layers.multiply() 函数用于执行输入数组的逐元素乘法。 得到新的权重数据。使得网络更加专注主要特征。
    # 为什么没有进行BN层，因为BN层强型将均值为0，方差为1，会有一些特征信息的数据丢失。
    x = layers.Multiply(name=prefix + 'se/multiply')([inputs, x])
    return x


"""
更新了Block，加入了SE机制，以及替换了激活函数
"""


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


def _inverted_res_block(x, input_channel: int, kernel_size: int, expend_channel: int,
                        output_channel, is_useSe: bool, activation: str,
                        strides: int, block_id: int, alpha: float = 1.0):
    bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)

    input_channel = _make_divisible(input_channel * alpha)
    expend_channel = _make_divisible(expend_channel * alpha)
    output_channel = _make_divisible(output_channel * alpha)
    activation = layers.ReLU if activation == "RE" else HardSwich
    shortcut = x
    prefix = 'expanded_conv'
    if block_id:
        # 根据blockId的编号来进行channel的升维
        # 第一个bneck是没有经过升维的。因为跟输入的维度是一样的。
        prefix = "expanded_conv_{}".format(block_id)
        x = layers.Conv2D(filters=expend_channel, kernel_size=1, padding='same', use_bias=False,
                          name=prefix + 'expand')(x)
        x = bn(name=prefix + 'expand/bn')(x)
        x = activation(name=prefix + 'expand/' + activation.__name__)(x)
    if strides == 2:
        # 步长为2时，进行补0操作，维度对齐
        input_size = (x.shape[1], x.shape[2])
        x = layers.ZeroPadding2D(padding=correct_pad(input_size, kernel_size), name=prefix + 'dw/pad')(x)
    #   如果步长为1则补0，否则valid
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same' if strides == 1 else 'valid',
                               use_bias=False, name=prefix + 'depthwise')(x)
    x = bn(name=prefix + 'depthwise/batchnorm')(x)
    x = activation(name=prefix + 'depthswise/' + activation.__name__)(x)
    if is_useSe:
        x = _se_block(x, filters=expend_channel, prefix=prefix)
    x = layers.Conv2D(filters=output_channel, kernel_size=1, padding='same', use_bias=False, name=prefix + 'project')(x)
    x = bn(name=prefix + 'project/bn')(x)
    # 进行残差
    return layers.Add(name=prefix + 'Add')([shortcut, x]) if strides == 1 and input_channel == output_channel else x


def MyMobileNetV3_large(input_shape=(224, 224, 3), num_class=4, alpha=1.0, include_top=True):
    bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)
    # 第一个是conv2d,16个channel
    image_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', use_bias=False, name='Conv')(image_input)
    x = bn(name='conv/bn')(x)
    x = HardSwich(name='conv/hardswish')(x)
    # 开始构建block
    inverted_cnf = partial(_inverted_res_block, alpha=alpha)
    """
    (x, input_channel, kernel_size, expend_channel,output_channel, is_useSe, activation,strides, block_id)
    """
    x = inverted_cnf(x, 16, 3, 16, 16, False, 'RE', 1, 0)
    x = inverted_cnf(x, 16, 3, 64, 24, False, 'RE', 2, 1)
    x = inverted_cnf(x, 24, 3, 72, 24, False, 'RE', 1, 2)

    x = inverted_cnf(x, 24, 5, 72, 40, True, 'RE', 2, 3)
    x = inverted_cnf(x, 40, 5, 120, 40, True, 'RE', 1, 4)
    x = inverted_cnf(x, 40, 5, 120, 40, True, 'RE', 1, 5)

    x = inverted_cnf(x, 40, 3, 240, 80, False, 'HS', 2, 6)
    x = inverted_cnf(x, 80, 3, 200, 80, False, 'HS', 1, 7)
    x = inverted_cnf(x, 80, 3, 184, 80, False, 'HS', 1, 8)
    x = inverted_cnf(x, 80, 3, 184, 80, False, 'HS', 1, 9)

    x = inverted_cnf(x, 80, 3, 480, 112, True, 'HS', 1, 10)
    x = inverted_cnf(x, 112, 3, 672, 112, True, "HS", 1, 11)
    x = inverted_cnf(x, 112, 5, 672, 160, True, "HS", 2, 12)
    x = inverted_cnf(x, 160, 5, 960, 160, True, "HS", 1, 13)
    x = inverted_cnf(x, 160, 5, 960, 160, True, "HS", 1, 14)

    con12_channel = _make_divisible(960 * alpha)
    conv13_channel = _make_divisible(1280 * alpha)
    x = layers.Conv2D(filters=con12_channel,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name="Conv_1")(x)
    x = bn(name="Conv_1/BatchNorm")(x)
    x = HardSwich(name="Conv_1/HardSwish")(x)
    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        #   因为全局平均池化会flatten，所以需要reshape
        x = layers.Reshape((1, 1, con12_channel))(x)
        #   一般全局平均池化后就softmax了，但是这里再经过了两个conv，并且直接经过了线型输出，这应该算是他的一个特点
        # fc1
        x = layers.Conv2D(filters=conv13_channel,
                          kernel_size=1,
                          padding='same',
                          name="Conv_2")(x)
        x = HardSwich(name="Conv_2/HardSwish")(x)

        # fc2
        x = layers.Conv2D(filters=num_class,
                          kernel_size=1,
                          padding='same',
                          name='Logits/Conv2d_1c_1x1')(x)
        x = layers.Flatten()(x)
        x = layers.Softmax(name="Predictions")(x)
        return Model(image_input, x, name='mobileNetV3_large')


if __name__ == "__main__":
    model = MyMobileNetV3_large(num_class=4)
    model.build([4, 224, 224, 3])
    print(model.summary())
