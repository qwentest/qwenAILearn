# coding: utf-8 
# @时间   : 2022/1/14 3:24 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : train.py
# @微信   ：qwentest123
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras import Model
import time, json, os

from model import MyGoogleNetV1

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    path = os.path.abspath(os.path.dirname(os.getcwd()))
    train_dir = "/Users/qcc/PycharmProjects/dataSet/train"
    var_dir = "/Users/qcc/PycharmProjects/dataSet/val"
    img_height = 224
    img_width = 224
    epochs = 10
    batch_size = 32

    # 处理数据
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

    model = MyGoogleNetV1(num_class=len(class_indices))
    model.build((batch_size, 224, 224, 3))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy']
                  )
    if not os.path.exists("./weights"): os.makedirs("./weights")
    # 该回调函数将在每个epoch后保存模型到filepath
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./weights/googleNetV1_{}.h5'.format(epochs),
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_loss')]
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
    t1 = time.time()
    main()
    print("训练时间={}".format(time.time() - t1))
