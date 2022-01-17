# coding: utf-8 
# @时间   : 2022/1/17 3:08 下午
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


class ConvBNRelu(layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding='same', **kwargs):
        super(ConvBNRelu, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=filters, strides=strides, kernel_size=kernel_size, padding=padding,
                                  use_bias=False,
                                  name='conv1')
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn')
        self.relu = layers.ReLU()

    def call(self, inputs, training=False, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.relu(x)


class DWConvBN(layers.Layer):
    def __init__(self, kernel_size, strides, padding, **kwargs):
        super(DWConvBN, self).__init__(**kwargs)
        self.dwconv = layers.DepthwiseConv2D(kernel_size=kernel_size, padding=padding, use_bias=False, strides=strides)
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='dw1')

    def call(self, inputs, training=False, **kwargs):
        x = self.dwconv(inputs)
        x = self.bn(x)  # 在v2.0中的block中,dw卷积之后，直接线性输出，没有经过relu
        return x


class ChannelShuffle(layers.Layer):
    """
    通道混洗。加强信息之间的沟通。其实现的方法，reshape,transpose,reshape
    """

    def __init__(self, shape, groups=2, **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        batch_size, height, width, num_channels = shape
        # shuffle时，需要指定分组的数量。分组的数量一定要是2的倍数
        assert num_channels % 2 == 0
        channel_per_group = num_channels // groups

        self.reshape1 = layers.Reshape((height, width, groups, channel_per_group))
        self.reshape2 = layers.Reshape((height, width, num_channels))

    def call(self, inputs, **kwargs):
        x = self.reshape1(inputs)
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        x = self.reshape2(x)
        return x


class ChannelSplit(layers.Layer):
    """
    当s = 1时，一分为2个分支，一个分支进行1*1,dw 3* 3,1*1的卷积操作，另一个保留原始信息，然后两者concat
    """

    def __init__(self, num_split: int = 2, **kwargs):
        super(ChannelSplit, self).__init__(**kwargs)
        self.num_splits = num_split

    def call(self, inputs, **kwargs):
        b1, b2 = tf.split(inputs, num_or_size_splits=self.num_splits,axis=-1)
        return b1, b2


def shuffle_block_s1(inputs, output_channel: int, strides: int, prefix: str):
    assert strides == 1
    assert output_channel % 2 == 0
    branch_c = output_channel // 2  # 因为要split，一半半channel
    x1, x2 = ChannelSplit(name=prefix + "/split")(inputs)

    x2 = ConvBNRelu(filters=branch_c, kernel_size=1, strides=1, name=prefix + "/x2_conv1")(x2)
    x2 = DWConvBN(kernel_size=3, strides=strides, padding='same', name=prefix + '/x2_dw1')(x2)  # 注意，这里没有经过relu
    x2 = ConvBNRelu(filters=branch_c, kernel_size=1, strides=1, name=prefix + "/x2_conv2")(x2)

    x = layers.Concatenate(name=prefix + "/concat")([x1, x2])  # v2中是concate之后再进行shuffle
    x = ChannelShuffle(x.shape, name=prefix + "/channel_shuffle")(x)
    return x


def shuffle_block_s2(inputs, output_channel: int, strides: int, prefix: str):
    assert strides == 2
    assert output_channel % 2 == 0
    branch_c = output_channel // 2  # 因为有2个分支，//2然后concate就跟输入的一致了

    # x1,先经过一个3*3的dw,再经过1*1的conv
    x1 = DWConvBN(kernel_size=3, strides=2, padding='same', name=prefix + "x1_s2/dw1")(inputs)
    x1 = ConvBNRelu(filters=branch_c, kernel_size=1, strides=1, name=prefix + "x1_s2/conv1")(x1)

    x2 = ConvBNRelu(filters=branch_c, kernel_size=1, strides=1, name=prefix + "/x2_conv1")(inputs)
    x2 = DWConvBN(kernel_size=3, strides=strides, padding='same', name=prefix + "/x2_dw1")(x2)
    x2 = ConvBNRelu(filters=branch_c, kernel_size=1, strides=1, name=prefix + "/x2_conv2")(x2)

    x = layers.Concatenate(name=prefix + "/s2_concate")([x1, x2])
    # 然后通道混洗
    return ChannelShuffle(x.shape, name=prefix + "/s2/channelShuffle")(x)


"""
根据网络结构图来构建
"""


def shuffleNet_v2(num_class: int, input_shape, stages_repeats: list, stages_out_channel: list):
    img_input = layers.Input(shape=input_shape)
    x = ConvBNRelu(filters=stages_out_channel[0],
                   kernel_size=3,
                   strides=2,
                   name="conv1")(img_input)

    x = layers.MaxPooling2D(pool_size=(3, 3),
                            strides=2,
                            padding='same',
                            name="maxpool")(x)
    #
    stage_name = ["stage_{}".format(i) for i in [2, 3, 4]]
    # stages_repeats=[4, 8, 4]
    # stages_out_channel = [24, 116, 232, 464, 1024]
    items = zip(stage_name,
                stages_repeats,
                stages_out_channel[1:])  # 因为24是第一个conv的channel
    for name, repeats, output_channels in items:
        for i in range(repeats):
            if i == 0:
                # shuffle_block_s2(inputs, output_channel: int, strides: int, prefix: str)
                x = shuffle_block_s2(x, output_channel=output_channels, strides=2, prefix=name + "_{}".format(i))
            else:
                x = shuffle_block_s1(x, output_channel=output_channels, strides=1, prefix=name + "_{}".format(i))
    # 有一个conv5
    x = ConvBNRelu(filters=stages_out_channel[-1],
                   kernel_size=1,
                   strides=1,
                   name="conv5")(x)
    x = layers.GlobalAveragePooling2D(name='globalpool')(x)

    x = layers.Reshape((1, 1, stages_out_channel[-1]))(x)
    # 修改为2个1 * 1的卷积，再直接线性输出
    x = layers.Conv2D(filters=stages_out_channel[-1],
                      kernel_size=1,
                      padding='same',
                      name="last_Conv_1")(x)
    x = layers.ReLU(max_value=6)(x)

    # fc2
    x = layers.Conv2D(filters=num_class,
                      kernel_size=1,
                      padding='same',
                      name='Logits/Conv2d_1c_1x1')(x)
    x = layers.Flatten()(x)

    # x = layers.Dense(units=num_class, name='fc')(x)  # 感觉可以像moblienetv3一样，在这里加2个1*1的卷积，然后直接线性输出
    x = layers.Softmax()(x)
    model = Model(img_input, x, name="ShuffleNetV2")
    return model


def shufflenet_v2_x1_0(num_class=4, input_shape=(224, 224, 3)):
    model = shuffleNet_v2(num_class=num_class,
                          input_shape=input_shape,
                          stages_repeats=[4, 8, 4],
                          stages_out_channel=[24, 116, 232, 464, 1024])
    return model


def shufflenet_v2_x0_5(num_classe=4, input_shape=(224, 224, 3)):
    model = shuffleNet_v2(num_class=num_classe,
                          input_shape=input_shape,
                          stages_repeats=[4, 8, 4],
                          stages_out_channel=[24, 48, 96, 192, 1024])
    return model


def shufflenet_v2_x2_0(num_classe=4, input_shape=(224, 224, 3)):
    model = shuffleNet_v2(num_class=num_classe,
                          input_shape=input_shape,
                          stages_repeats=[4, 8, 4],
                          stages_out_channel=[24, 244, 488, 976, 2048])
    return model


if __name__ == "__main__":
    model = shufflenet_v2_x1_0(num_class=4)
    model.build([4, 224, 224, 3])
    print(model.summary())
