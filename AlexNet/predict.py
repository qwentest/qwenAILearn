# coding: utf-8 
# @时间   : 2022/1/13 6:56 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : predict.py
# @微信   ：qwentest123
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, MaxPool2D
from tensorflow.keras import Model
import json
from PIL import Image
from model import MyAlextNet


def main(img_path):
    img_height = 227
    img_width = 227
    img = Image.open(img_path)
    img = img.resize((img_width, img_height))

    img = np.array(img) / 255.
    # 表示在a的第一个维度上增加一个新的维度，而其他维度整体往右移，最终得到shape为(1, m, n, c)的新数组
    img = (np.expand_dims(img, 0))
    class_indices = {"0": "daisy", "1": "dandelion", "2": "roses", "3": "sunflowers", "4": "tulips"}
    model = MyAlextNet(num_classes=len(class_indices))
    model.build((1,227,227,3))
    weight_path = "./weights/myAlex.h5"
    model.load_weights(weight_path)

    result = model.predict(img)
    # np.squeeze这个函数的作用是去掉矩阵里维度为1 的维度。例：(1, 300)的矩阵经由np.squeeze处理后变成300;
    result = np.squeeze(result)
    predict_class = np.argmax(result)
    # 预测的结果
    res = "class: {}   prob: {:.2}".format(class_indices[str(predict_class)],
                                           result[predict_class])
    print(res)


if __name__ == "__main__":
    main('./test/1.jpeg')
