# lim的博客

爱数据，爱生活，致力于做一名数据搬运工。  
热衷数据科学，机器学习，和深度学习应用，希望一切会越来越好。
在这里分享和总结一些自己平时积累的零碎知识和参加的一些竞赛题。  
这里，博主主要通过jupyter notebook的交互方式，结合注释来进行代码的编写和图的展示。

## Kaggle竞赛
### Titanic
相信对kaggle有一定了解和初入数据领域的同学们都做过titanic竞赛。
titanic竞赛是一个分类问题，参赛同学通过对训练集中每位乘客的年龄、票等级、家庭信息等情况进行分析，
判断测试集中每位乘客的存活情况，最后使用accuracy作为标准进行评价。虽然这是一个入门的竞赛题，
网上也非常多的参考，但博主还是有很认真的做（前前后后花了1个多月的时间，虽然成绩还是有点不满意），
最高成绩是排在Top6%左右。  
[TITANIC](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/Titanic%20Analysis.ipynb)

## 深度学习小游戏
### 入门之手写数字识别
相信接触过深度学习的童鞋对手写数字识别的任务并不陌生。它一个入门的阶段，是每位初学深度学习的同学的基本学习任务。
任务的目标是建立一个分类模型，对0-9的黑白手写数字图片进行识别。
这里使用了两种方法：  
1.使用CNN对训练集做特征提取，然后最后一层使用逻辑回归做预测；
2.训练RNN模型，对测试集做预测，将图片的每一行作为一个时序step，每一行的每一列作为一个时序step的特征；
以上两种方法均使用了pyTorch库。这里提一下pyTorch，个人认为相对Tensorflow，pyTorch更加轻量级，python接口友好，适合深度学习应用的快速实现。
文档齐全，适合入门的同学[pyTorch](http://pytorch.org/)。
[MNIST-CNN](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/MNIST%20Recognize.ipynb)  
[MNIST-RNN]()

## Spark
对Spark的应用主要是在公司的大大小小的项目中，因此这部分可能没太多的代码展示，
目前博主接触过的最大数据量的项目为联通集团的数据挖掘项目。



## others...
### 热力地图展示

