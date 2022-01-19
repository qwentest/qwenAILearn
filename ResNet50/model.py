# coding: utf-8 
# @时间   : 2022/1/15 9:01 上午
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

"""
    基本的残差结构，即两个3 * 3的卷积后相加
"""


class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        """
        :param out_channel: 上一个节点输出的通道数
        :param strides: 默认步长是1
        :param downsample: 当上一个节点输出通道数与当前通道数不一致时，需要通过一个1x1的卷积下采样
        :param kwargs:
        """
        super(BasicBlock, self).__init__(**kwargs)

        # 基本残差中的3*3的卷积中的channel是相同的
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides, padding='same', use_bias=False)
        """
        axis: 整数，指定要规范化的轴，通常为特征轴。例如在进行data_format="channels_first的2D卷积后，一般会设axis=1。
        momentum: 动态均值的动量
        epsilon：大于0的小浮点数，用于防止除0错误
        center: 若设为True，将会将beta作为偏置加上去，否则忽略参数beta
        scale: 若设为True，则会乘以gamma，否则不使用gamma。当下一层是线性的时，可以设False，因为scaling的操作将被下一层执行。
        beta_initializer：beta权重的初始方法
        gamma_initializer: gamma的初始化方法
        moving_mean_initializer: 动态均值的初始化方法
        moving_variance_initializer: 动态方差的初始化方法
        beta_regularizer: 可选的beta正则
        gamma_regularizer: 可选的gamma正则
        beta_constraint: 可选的beta约束
        gamma_constraint: 可选的gamma约束
        参考：https://keras-cn.readthedocs.io/en/latest/layers/normalization_layer/
        """
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        ####
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False, **kwargs):
        identity = inputs
        # 是否需要下采样改变通道数
        if self.downsample is not None:
            identity = self.downsample(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.add([identity, x])  # 需要注意的是，是加了之后再relu的
        x = self.relu(x)
        return x


"""
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
"""


class BottleNeck(layers.Layer):
    expansion = 4

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BottleNeck, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        ######
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False, strides=strides, padding='same')
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        ######最后一个1x1的channel刚好是输入channel的4倍
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        ######
        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()

    def call(self, inputs, training=False, **kwargs):
        identity = inputs
        # 是否需要下采样改变通道数
        if self.downsample is not None:
            identity = self.downsample(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.add([x, identity])  # 需要注意的是，是加了之后再relu的
        x = self.relu(x)
        return x


class MyResNet(Model):
    def __init__(self, block, block_num, num_class=4, include_top=True, **kwargs):
        super(MyResNet, self).__init__(**kwargs)
        self.include_top = include_top
        # 在进入残差前有一个7 * 7的卷积，以及一个池化操作
        self.conv1 = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu1 = layers.ReLU()
        self.maxpool1 = layers.MaxPool2D(pool_size=3, strides=2, padding='same')


        # 构建残差块 NHWC，只有第一个残差块的s=1,其它都是s=2
        self.block1 = self._make_layer(block, True, 64, block_num[0])
        self.block2 = self._make_layer(block, False, 128, block_num[1], strides=2)
        self.block3 = self._make_layer(block, False, 256, block_num[2], strides=2)
        self.block4 = self._make_layer(block, False, 512, block_num[3], strides=2)

        if self.include_top:
            self.avgpool = layers.GlobalAvgPool2D()  # 全局平均池化
            self.fc = layers.Dense(num_class)
            self.softmax = layers.Softmax()

    def _make_layer(self, block, isFirstBlock, channel, block_num, strides=1):
        downsample = None
        # 当strides为2或者输入chnnael与输出chnnael不一致时，需要通过一个1*1的卷积将维
        if strides != 1 or isFirstBlock is True:
            downsample = Sequential([
                layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides, use_bias=False),
                layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
            ])
        # 先将维
        layers_list = []
        layers_list.append(block(channel, downsample=downsample, strides=strides))
        # 根据结构图，重复残差结构多少次
        for index in range(1, block_num):
            layers_list.append(block(channel))
        return Sequential(layers_list)

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)

        if self.include_top:
            x = self.avgpool(x)
            x = self.fc(x)
            x = self.softmax(x)

        return x


# resnet50用的是bottleNeck即倒残差模块
def resnet50(num_class=4, include_top=True):
    block = BottleNeck
    return MyResNet(block, [3, 4, 6, 3], num_class, include_top)


if __name__ == "__main__":
    model = resnet50()
    model.build([4, 224, 224, 3])
    print(model.summary())
