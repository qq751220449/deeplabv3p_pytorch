# DeepLabv3+模型复现

复现模型是基于百度的阿波罗数据集进行语义分割模型的复现，下载百度阿波罗数据放入`dataset`目录。

## 环境

- Ubuntu18.04
- Python3.6.5
- torch==1.5.0
- torchvision==0.6.0
- GTX 1080 - Nvidia



### 支持的Backbone

目前支持`xception`、`atros_resnet`两种特征提取网络，具体可以在`config.py`文件中进行设置。支持`step`，`poly`，`cos`等学习率调整方式，详细见`config.py`文件




### 训练

Run `train.py` by

```sh
python train.py
```
