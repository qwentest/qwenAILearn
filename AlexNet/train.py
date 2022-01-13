# coding: utf-8 
# @时间   : 2022/1/13 8:26 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : train.py
# @微信   ：qwentest123
import json
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, MaxPool2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from model import MyAlextNet
import os


def main():
    path = os.path.abspath(os.path.dirname(os.getcwd()))
    train_dir = "/Users/qcc/PycharmProjects/dataSet/train"
    var_dir = "/Users/qcc/PycharmProjects/dataSet/val"
    img_height = 227
    img_width = 227
    epochs = 10
    batch_size = 32
    """
    ImageDataGenerator()是keras.preprocessing.image模块中的图片生成器，同时也可以在batch中对数据进行增强，扩充数据集大小，增强模型的泛化能力。比如进行旋转，变形，归一化等等。
    参考：https://www.jianshu.com/p/d23b5994db64
    """
    train_image_generator = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
    val_image_generator = ImageDataGenerator(rescale=1. / 255)
    # 从路径生成增强数据
    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(img_height, img_width),
                                                               class_mode='categorical'
                                                               )
    total_train = train_data_gen.n

    val_data_gen = val_image_generator.flow_from_directory(directory=var_dir,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           target_size=(img_height, img_width),
                                                           class_mode='categorical'
                                                           )

    total_val = val_data_gen.n
    # 将文件夹的名称生成key,value是编号
    class_indices = train_data_gen.class_indices
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # 将类别写到json文件中，供pre的时候用
    with open("{}/class_indices.json".format(path), 'w') as f:
        f.write(json.dumps(inverse_dict))

    model = MyAlextNet(num_classes=len(inverse_dict))
    model.build((batch_size,227,227,3))
    model.summary()
    # training
    # model.load_weights("./weights/myAlex.h5")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy']
                  )
    # 该回调函数将在每个epoch后保存模型到filepath
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./weights/myAlex.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_loss')]
    # 135011 // 32 = 4219
    # 524 // 32 = 16
    history = model.fit(x=train_data_gen,
                        steps_per_epoch=total_train // batch_size,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,
                        callbacks=callbacks)

    # 训练过程中的loss和accuracy做一个图像的可视化
    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == "__main__":
    main()
