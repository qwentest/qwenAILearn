# coding: utf-8 
# @时间   : 2022/1/12 1:33 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 训练代码
# @文件   : train.py
# @微信   ：qwentest123
import time

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam,SGD
# 引用自己定义的网络结构
from model import MyLeNet5
from datetime import datetime


def main():
    model = MyLeNet5()
    # 定义损失函数
    """
    交叉熵损失，参考：https://blog.csdn.net/qq_37297763/article/details/106169857
    from_logits = True，即网络层输出为Linear;
    from_logits = False，代表网络层输出为softmax;
    reduction = 'auto'，代表最后会求平均
    """
    loss_object = SparseCategoricalCrossentropy()
    # 定义优化器,使用默认参数
    optimizer = Adam(learning_rate=0.03)

    """
    参考： https://blog.csdn.net/qq_39507748/article/details/105267168
    在2.0中，metrics是用来在训练过程中监测一些性能指标，而这个性能指标是什么可以由我们来指定
    
    参考用法：https://blog.csdn.net/jpc20144055069/article/details/105324654
    计算给定值的（加权）平均值
    """
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name='test_accuracy')

    """
    自定义训练方法
    tf.GradientTape是官方大力推荐的用法,用来自动训练梯度
    使用方法参考： https://zhuanlan.zhihu.com/p/146016883
    f.function本质上构造一个可调用的可执行程序，该可调用程序执行通过在func中跟踪编译TensorFlow操作而创建TensorFlow图（tf.Graph），从而有效地将func作为TensorFlow图执行
    参考：https://zhuanlan.zhihu.com/p/350746893
    """

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            # 计算loss
            loss = loss_object(labels, predictions)
        # 自动求梯度
        gradients = tape.gradient(loss, model.trainable_variables)
        # 把计算出的梯度更新到变量中去
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # 训练过程参数性能指标的监视
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        # 监视预测过程参数的性能指标
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    """
    数据集的下载与处理
    """
    mnist = tf.keras.datasets.mnist
    #   下载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #   归一化处理
    x_train, x_test = x_train / 255.0, x_test / 255.0

    """
    tf.newaxis,增加一个维度
    参考： https://blog.csdn.net/llabview/article/details/120196249
    """
    x_train = x_train[..., tf.newaxis]  # tensorflow输入的数据格式是(图片数，长，宽，通道），这里加1维是因为图片本身为灰度图
    x_test = x_test[..., tf.newaxis]

    """生成数据"""
    # from_tensor_slices,它的作用是把给定的元组、列表和张量等数据进行特征切片
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    EPOCHS = 5
    for epoch in range(EPOCHS):
        # 重置指标的值
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        # 训练集
        for images, labels in train_ds:
            train_step(images, labels)
        # 验证集
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Time {}, Epoch {}, Train_Loss: {}, Train_Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(datetime.now(),
                              epoch + 1,
                              train_loss.result(),
                              train_accuracy.result(),
                              test_loss.result(),
                              test_accuracy.result()))
        if True if epoch % 5 == 0 else None:
            model.save_weights('./weights/mnist')


if __name__ == "__main__":
    main()
