# 数学口算批改实现-检测部分 

基于RetinaNet目标检测方案实现的数学口算题批改（试题检测部分）, 参考keras-retinanet实现


## 安装

参考[keras-retinanet](https://github.com/fizyr/keras-retinanet)

## Training

* 数据准备，train.csv文件(test.csv 相同)，格式如下：
 
 ```
 /path/for/data_001.jpg,x1,y1,x2,y2,class1
 /path/for/data_001.jpg,x1,y1,x2,y2,class2
 /path/for/data_002.jpg,x1,y1,x2,y2,class2
 /path/for/data_002.jpg,x1,y1,x2,y2,class3
 
 ``` 
* 数据准备，标签文件 names.csv

```
class1,0
class2,1
class3,2
class4,3
class5,4

```
*  训练脚本：
相关参数可根据实际情况调整

```
python keras_retinanet/bin/train.py \
 --random-transform --gpu 1 \
 --epochs 30 \
 --snapshot-path ./snapshots \
 --steps 20000 \
 csv ./train_data/train.csv ./train_data/names.csv \
 --val-annotations ./train_data/test.csv

```
## 转换成推理模型
训练好的模型需要经过一次转换，转为可以用来进行推理的模型:

```shell
# Running directly from the repository:
keras_retinanet/bin/convert_model.py /path/to/training/model.h5 /path/to/save/inference/model.h5

# Using the installed script:
retinanet-convert-model /path/to/training/model.h5 /path/to/save/inference/model.h5
```

Most scripts (like `retinanet-evaluate`) also support converting on the fly, using the `--convert-model` argument.


## 推理测试

```
python infer.py --model=/path/for/infer.h5 --imgpath=/path/for/img.jpg
```

