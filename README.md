# qwenAILearn
计算机视觉AI技术学习的代码记录

环境:macos12.0.1 + pycharm 2021.1.1 + python3.8 + tensorflow2.6 + annconda   

1.各个模型之间的简单性能对比(相同的分类数据，相同的学习率，相同的epoch)  

---  
vgg16       14714688    119566341   537.2MB    2762s    0.24     0.42s(预测错了)  
AlexNet     3747200     54554629    233.2MB    238s     0.24     0.12s(预测错了)  
GoogeNetv1  5880402     /           23.7MB     577s     0.52     0.51s        
ResNet50    23571397    /           94.5MB     1091s    0.70     0.52s  
MobileNetV1 2419013     /           9.9MB      386s     0.76     0.24s 
MobileNetV2 2230277     /           9.3MB      442s     0.77     0.33s   
MobileNetV3 4208437     /           17.5MB     490s     0.77     0.59s(预测错了)  
ShuffleNet2 1258729     /           5.6MB      276s     0.76     0.47s  
ShuffleNoFC 2308329     /           9.8MB      221s     0.79     0.47s  
DarkNet-19  16872874    /           67.7MB     508s     0.74     0.23s  
DarkNet-53  13284970    /           53.3MB     1053s    0.68     0.16s   
CSPDarkNet  26678260    /           107MB      太长了    0.74     1.04s   
EfficientV1 3996097     /           16.6MB     860s     0.74     0.59(预测错了)  
EfficientV2 20183893    /           82MB       1906s    0.22     1.32(预测错了)
---