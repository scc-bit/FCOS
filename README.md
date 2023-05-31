# FCOS

## 文件目录

FCOS

├── model                     //FCOS网络结构及设置

│   ├── backbone              //Resnet-50 Backbone  

│   │   └── resnet.py          //Resnet-50    

│   ├── fcos.py                  // fcos网络

│   ├── fpn.py                  // FPN结构

│   ├── head.py                  //Head结构

│   ├── loss.py                  //损失函数

│   ├── metric.py                  //评价函数

│   └── config.py 	         //模型设置

├── dataloader            //VOC加载

│  └── VOC_dataset.py	//加载VOC数据

├── VOCdevkit                  //存放VOC数据集的文件夹，数据下载解压后放在这里

├── weights                  //存放训练好的模型（.pth文件）的文件夹 

├── logs                  //存放训练验证过程的loss、mAP值等的文件夹，tensorboard可视化用

├── test_images                  //测试demo的图片

│   ├── input              //存放待测试的图片 

│   └── output         //存放demo测试后的图片结果 

├── train_and_eval.py                 //训练模型和验证模型，输出的结果会存放到weights、logs文件夹（！速度较慢）

├── train.py                 //训练模型，输出的结果会存放到weights、logs文件夹

├── eval.py                 //加载训练好的模型（.pth文件），可以得到AP和mAP结果

├── demo.py                 //加载测试图片，可以得到目标检测结果

└── requirements.txt                  //环境依赖





## 示例命令 

```
cd FCOS

pip install requirements.txt   
```

### 训练

```
python train_and_eval.py #速度较慢
```
或

```
python train.py  #速度比同时train和evaluate要快
```

默认30个epochs，程序将默认保存每个epoch后的模型 .pth文件

训练完成后可以用tensorboard查看记录的loss

```
tensorboard --logdir=FCOS/logs
```

### 验证

```
python eval.py  
```

导入训练后得到的.pth文件 ，并在验证集上测试得到所有class的AP和mAP

### 测试

```
python demo.py 
```

导入训练后得到的.pth文件 ，并在测试图片上测试得到目标检测结果

### 数据集

VOC2012：

http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

### Backbone

Resnet-50：
https://download.pytorch.org/models/resnet50-19c8e357.pth
