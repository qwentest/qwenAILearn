# coding: utf-8 
# @时间   : 2022/1/13 8:20 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : readH5.py
# @微信   ：qwentest123
import h5py

f = h5py.File('./AlexNet/weights/myAlex.h5', 'r')
for root_name, g in f.items():
    for _, weights_dirs in g.attrs.items():
        for i in weights_dirs:
            name = root_name + "/" + str(i, encoding="utf-8")
            data = f[name]
            print(data.value)
