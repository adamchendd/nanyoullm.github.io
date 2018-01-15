# lim力铭的简易博客

> 爱数据，爱生活，云吸猫患者，坐标深圳。  
热爱数据科学，机器学习，和深度学习应用。
在这里分享和总结一些自己平时积累的零碎知识和做过的一些小应用。  
博主主要通过jupyter notebook结合md的交互方式，结合注释来进行代码的编写和图的展示。  
[仓库](https://github.com/nanyoullm/nanyoullm.github.io)  
[主页](https://nanyoullm.github.io/)  

## Kaggle
> kaggle有许多关于数据科学、机器学习的竞赛，是一个交流分享、锻炼技能的绝佳平台。

### Titanic
> 相信对kaggle有一定了解和初入数据领域的同学们都做过titanic竞赛。
titanic竞赛是一个分类问题，参赛同学通过对训练集中每位乘客的年龄、票等级、家庭信息等情况进行分析，
判断测试集中每位乘客的存活情况，最后使用accuracy作为标准进行评价。虽然这是一个入门的竞赛题，
网上也非常多的参考，但博主还是有很认真的做（前前后后花了1个多月的时间，虽然成绩还是有点不满意），
最高成绩是排在Top6%左右。  
[TITANIC](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/Titanic%20Analysis.ipynb)

### Give me some credit
> 待续

## 机器学习
> 这里主要是自己在之前在学习machine learning过程中做的一些笔记和总结，以方便自己反刍。
总结里部分是在学习李航老师的《统计学习方法》中做的笔记，所以有不少对应内容是摘自书中，在此说明。
另有部分是从其他学习途径总结的。

### 机器学习数学基石
> 之前参加了《机器学习数学基石课程》，也是对毕业前学习的课程上做了些补充，
这里回顾一下机器学习优化问题的知识点。  
[优化问题](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B9%8B%E4%BC%98%E5%8C%96%E9%97%AE%E9%A2%98.ipynb)

### 统计学习方法概论
> 这里主要叙述统计学习方法的一些基本概念。  
[概论](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E6%A6%82%E8%AE%BA.ipynb)

### 感知机
> 感知机是二类分类的线性分类模型，输入为实力的特征向量，输出为实例的类别，是判别模型。
感知机学习旨在求出将训练数据进行线性划分的分离超平面，是神经网络和支持向量机的基础。  
这是博主接触的第一个ml算法，至今还记得导师布置的用感知机提取红色印章的小作业。  
[感知机](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/%E6%84%9F%E7%9F%A5%E6%9C%BA.ipynb)

### k-nn远亲不如近邻
> k近邻（k-nearest neighbor）是一种基本的分类和回归方法。
分类时，对新的实例，根据其k个最近邻的训练实例的类别，通过多数表决等方式进行预测。
因此，k近邻法不具有显式的学习过程。  
k近邻实际上利用训练数据集对特征向量空间进行划分，并作为其分类的模型。  
[k-nn](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/k-nn.ipynb)

### 朴素贝叶斯
> 朴素贝叶斯是基于贝叶斯定理和特征条件独立的分类方法，对于给定的训练集，
基于特征条件独立假设，学习输入/输出的联合概率分布，然后基于此分布，
对给定的x，利用贝叶斯定理求出后验概率最大的输出y。  
[bayes](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/Bayes.ipynb)

### 决策树
> 决策树是一种基本的分类与回归方法。在其分类问题中，表示基于特征对实例进行分类的过程，可以认为是
if-else规则的集合，也可以认为是定义在特征空间与类空间上的条件分类。其主要优点是模型具有可读性、分类速度快。
[决策树]()

## 深度学习
> 深度学习是一门相当系统和完整的知识体系，涉及到的知识点非常多，在这里博主主要分享一些做过的应用，和一些容易忽略的细节点。使用到的库主要为pyTorch。

### pyTorch基础
> 个人觉得pyTorch是一个很好用的计算框架，使用深度学习的快速开发。  
PyTorch 由 Adam Paszke、Sam Gross 与 Soumith Chintala 等人牵头开发，其成员来自 Facebook FAIR 和其他多家实验室。它是一种 Python 优先的深度学习框架，在今年 1 月被开源，提供了两种高层面的功能：
使用强大的 GPU 加速的 Tensor 计算（类似 numpy）
构建于基于 tape 的 autograd 系统的深度神经网络。  
这里[pyTorch基础](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/pyTorch_basic.ipynb)
是我学习官网教程后写的一些快速入门总结，欢迎参考。

### 基石logistic regression
> 待续

### 炼丹提取CNN
> 参考http://blog.csdn.net/u014114990/article/details/51125776

### 地转轮回RNN
> 我们知道，人类思考的时候并不是从一片空白的大脑开始的（不像只有7秒记忆的鱼），而是会参考之前的经验和阅历。就像我们看悬疑片的时候，大脑会回溯之前的
种种细节；阅读文章的时候，会基于对先前的理解来对文章主旨做出判断。  
而这恰恰是传统的神经网络做不到的。地转轮回的RNN循环神经网络的出现解决了这个问题，它能记忆前时序的信息，其类似CNN的共享参数机制也是极大简化了网络的复杂性。
由RNN进化的LSTM和GRU单元更是展现了神通广大的本领。  
[RNN-LSTM](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/Basic-RNN.ipynb)

### 不要怂就是GAN
> 待续

### 入门之手写数字识别
> 接触过深度学习的同学对手写数字识别的任务并不陌生。它一个入门的阶段，是每位初学深度学习的同学的基本学习任务。
任务的目标是建立一个分类模型，对0-9的黑白手写数字图片进行识别。
这里使用了两种方法：  
1.使用CNN对训练集做特征提取，然后最后一层使用逻辑回归做预测；  
2.训练RNN模型，对测试集做预测，将图片的每一行作为一个time step，每一行的每一列作为一个时序step的特征；  
以上两种方法均使用了pyTorch库。这里提一下pyTorch，个人认为相对Tensorflow，pyTorch更加轻量级，python接口友好，适合深度学习应用的快速实现。
文档齐全，适合入门的同学[pyTorch](http://pytorch.org/)。  
[MNIST-CNN](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/MNIST%20Recognize.ipynb)  
[MNIST-RNN](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/MNIST-RNN.ipynb)

### Transfer Learning站在巨人的肩膀上
> 待续

### neural style
> 待续

### pyTorch & tensorborad
> 待续

## Spark
> 我们身处于大数据的一个起步时代，作为一个码农，掌握大数据计算平台的使用还是必须的。
Apache Spark作为近年来最流行的大数据开源项目，可以说是占据了大小互联网的生产环境，
是一个“快如闪电的集群计算”工具。  
关于Spark的介绍我也不多说啦，可以看看官网[Spark](https://spark.apache.org/)。
这里需要说明的是，博主并没有阅读过Spark的源码，对Spark较多的是停留在应用层面，
博客的内容很多是给自己回顾所涉及的点，不是很完善。
所述之处如有错误，还请及时指出。
对Spark的应用主要是在公司的大小日常项目中，目前博主接触过的最大数据量的项目为联通集团的数据挖掘项目。

### Spark基础回顾
> [Spark基础回顾](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/spark%E4%B9%8B%E5%9F%BA%E7%A1%80.ipynb)

### Spark & jupyter & Scala 环境搭建
> 为了方面展示，搭建了spark & jupyter环境，使用的scala语言内核。  
scala环境搭建参考：http://blog.csdn.net/he582754810/article/details/53837142  
jupyter spark kernel搭建参考：http://blog.csdn.net/u012948976/article/details/52372644

### 数据加工：RDD基本操作
> part1: [单词统计](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/spark%E5%8D%95%E8%AF%8D%E7%BB%9F%E8%AE%A1.ipynb)  
part2: 用户通话记录多维度统计

### MLlib
> 待续

### GraphX
> 待续

## Python Web

### Flask
> Flask 是一个 Python 实现的 Web 开发微框架。对于初次接触 web 应用开发的同学来说，
是很容易上手的。我是从官网的简单例子入手的，话不多说来看一看。  
[flask 之 hello world](http://docs.jinkan.org/docs/flask/quickstart.html#a-minimal-application)

### 模型稳定性监控
> 一个模型上线后，在实际生产中需要监控这个模型的稳定性的，一旦模型的
产出出现异常，一点点的偏差就会带来极大的风险损失。  
以金融风险评分为例，我们需要定期按照新观察期的数据，去更新每个用户的信用评分。
更新之后的评分和各变量的分布的分析是必要的，通过多账期评分总体的分布对比，
计算一些指标， 可以评价这个模型是否收到了当前业务的影响？
在一定周期内是否可以保持稳定？PSI是多少？
申请了金融产品的客户的分数是否满足排序性？整体KS是多少？等等...  
这里我写了一个简单 web 应用，用于监控模型分布的稳定性，方便我们做相关的数据分析。  
项目包含完整的前后端代码。详细内容及代码请跳转: [简易模型监控](https://github.com/nanyoullm/simple-model-monitor)

## 数据可视化

### seaborn
> seaborn是一个python可视化的工具，是基于 matplotlib ，能与 dataframe 数据结构有更好切合的工具。
[seaborn使用](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/seaborn_visualization.ipynb)

## 信用风险评分套路

### 评分卡的开发过程
> 这里介绍一下评分卡的开发的常规操作。它实际上是一个数据科学的课题，
流程上有很多与机器学习实际问题相同的地方，但是它又具有其特别之处，如具有时间周期性，特征业务可解释性的需求等...
让我们来认识一下吧！  
[评分卡的开发过程](https://github.com/nanyoullm/nanyoullm.github.io/blob/master/src/%E4%BF%A1%E7%94%A8%E9%A3%8E%E9%99%A9%E8%AF%84%E5%88%86%E5%8D%A1%E7%A0%94%E7%A9%B6/%E8%AF%84%E5%88%86%E5%8D%A1%E7%9A%84%E5%BC%80%E5%8F%91%E8%BF%87%E7%A8%8B.ipynb)

## Others...

### 量化平台之初体验
> ricequant平台使用

### 云吸猫...
> 会有猫的