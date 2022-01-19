# coding: utf-8 
# @时间   : 2022/1/12 9:17 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : LeNet5
# @文件   : model19.py
# @微信   : qwentest123

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, MaxPool2D
from tensorflow.keras import Model


class MyLeNet5(Model):
    def __init__(self):
        super(MyLeNet5, self).__init__()
        """
        参数解释：
        完整的可参考：https://blog.csdn.net/godot06/article/details/105054657
        keras.layers.convolutional.Conv2D(filters, # 卷积核数目
                                  kernel_size, # 过滤器的大小
                                  strides(1,1),  # 步长
                                  padding='valid', # 边界处理
                                  data_format=None, 
                                  dilation_rate=(1,1), 
                                  activation=None, # 激活函数
                                  use_bias=True, #是否使用偏置量,布尔值
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  kernel_regularizer=None,
                                  bias_regularizer=None,
                                  activity_regularizer=None,
                                  kernel_constraint=None,
                                  bias_constraint=None)
        """
        # 第一组卷积、池化
        self.conv1 = Conv2D(filters=6,
                            kernel_size=(5, 5),
                            activation='sigmoid',
                            input_shape=(28, 28, 1),  # 因为要求的是输入灰度度，所以通道数为1
                            padding='same'
                            )  # 得到28 * 28 * 6
        self.pool1 = AvgPool2D(pool_size=(2, 2), strides=2)  # 得到14 * 14 * 6

        # 第二组卷积、池化
        # 得到10 * 10 * 16, 这里应该vaild，如果是same那么尺寸不变
        self.conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid')
        self.pool2 = AvgPool2D(pool_size=(2, 2), strides=2)  # 得到5*5*16

        # 摊平所有数据
        self.flatten = Flatten()

        """
        参数解释
        tf.keras.layers.Dense(
            units,                                 # 正整数，输出空间的维数
            activation=None,                       # 激活函数，不指定则没有
            use_bias=True,						   # 布尔值，是否使用偏移向量
            kernel_initializer='glorot_uniform',   # 核权重矩阵的初始值设定项
            bias_initializer='zeros',              # 偏差向量的初始值设定项
            kernel_regularizer=None,               # 正则化函数应用于核权矩阵
            bias_regularizer=None,                 # 应用于偏差向量的正则化函数
            activity_regularizer=None,             # Regularizer function applied to the output of the layer (its "activation")
            kernel_constraint=None,                # Constraint function applied to the kernel weights matrix.
            bias_constraint=None, **kwargs         # Constraint function applied to the bias vector
        )
        """
        # 全连接
        self.fc1 = Dense(120, activation="sigmoid")
        self.fc2 = Dense(84, activation="sigmoid")
        self.fc3 = Dense(10, activation="softmax")#因为手写数字识别的类别个数是10

    """
    call()的本质是将一个类变成一个函数（使这个类的实例可以像函数一样调用）
    参考：https://blog.csdn.net/weixin_44207181/article/details/90648473
    """

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    x = tf.random.normal([1, 28, 28, 1])  # 4 张,28 * 28大小，1个通道的灰色图像
    model = MyLeNet5()
    model.build([1, 28, 28, 1])
    print(model.summary())
    # y = model(x)
    # print(y)
