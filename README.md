SVM、BP、C4.5实现iris数据集、optdigits数据集分类
========

数据集介绍
------
### iris
    150行数据，每行数据包括花瓣长度、花瓣宽度、花萼长度、花萼宽度四个连续型特征  
    共分三个种类：iris-setosa、iris-versicolor、iris-virginica
### optidigits：
    手写数字识别数据集，分为训练数据集optdigits_tra和检验数据集optdigits_tes
    每个数据有64个像素特征，分为10个种类：数字0，1，2，3，4，5，6，7，8，9
    
--------

运行环境介绍
------
  ptyhon3.9

使用库说明
-----
  math、random、panda、numpy、time、sklearn、matplotlib、operator、spicy.special、sklearn.model_selection、sklearn.datasets

--------

实验结果展示
------
### iris：

* iris_bp:
  accuracy | Running time
  ---------|------------
  0.979166667 |33.06088733673096 Seconds
  
* iris_svm:

  train prediction|test prediction|train Accuracy|test Accuracy|Running time
  ----------------|---------------|--------------|-------------|------------
  0.971           |0.978          |0.971         |0.978        |1.6316866874694824 Seconds
  
* iris_c4.5:
  accuracy | Running time
  ---------|------------
  0.9333333333333333 |0.008972406387329102 Seconds

### optdigits:
* optdigits_bp:  

                       precision    recall  f1-score   support  
     

                   0       0.64      0.99      0.78       178  
           
                   1       0.70      0.79      0.74       182  
           
                   2       0.84      0.85      0.84       177  
             
                   3       0.73      0.87      0.79       183  
             
                   4       0.86      0.96      0.91       181  
           
                   5       0.95      0.44      0.60       182  
           
                   6       0.84      0.89      0.86       181  
           
                   7       0.84      0.91      0.87       179  
            
                   8       0.90      0.45      0.60       174  
           
                   9       0.63      0.56      0.59       180  
           
            accuracy                           0.77      1797  
    
           macro avg       0.79      0.77      0.76      1797  
   
        weighted avg       0.79      0.77      0.76      1797  

  accuracy | Running time
  ---------|------------
  0.77 |2.7409827709198 Seconds
  
* optdigits_svm:
  
  train prediction|test prediction|train Accuracy|test Accuracy|Running time
  ----------------|---------------|--------------|-------------|------------
  0.990           |0.970          |0.990         |0.970        |2.512850046157837 Seconds
  
* iris_c4.5:

  accuracy | Running time
  ---------|------------
  0.8269337785197551 |4.550484657287598 Seconds
-------------------------------
算法比较
------
#### Svm
>优点：
* 使用核函数可以向高维空间进行映射
* 使用核函数可以解决非线性的分类
* 分类思想很简单，就是将样本与决策面的间隔最大化
* 分类效果较好
>缺点：
* 对大规模数据训练比较困难
* 无法直接支持多分类，但是可以使用间接的方法来做

#### BP：
>优点：
* 非线性映射能力。
* 自学习和自适应能力：BP神经网络在训练时，能够通过学习自动提取输出、输出数据间的“合理规则”，并自适应的将学习内容记忆于网络的权值中。
* 泛化能力较强。
* 容错能力：BP神经网络在其局部的或者部分的神经元受到破坏后对全局的训练结果不会造成很大的影响。
>缺点：
* BP 神经网络算法的收敛速度慢：由于BP神经网络算法本质上为梯度下降法，它所要优化的目标函数是非常复杂的，因此，必然会出现“锯齿形现象”，这使得BP算法低效。
* BP 神经网络结构选择不一：BP神经网络结构的选择至今尚无一种统一而完整的理论指导，一般只能由经验选定，例如隐藏层节点数。
* 应用实例与网络规模的矛盾问题：BP神经网络难以解决应用问题的实例规模和网络规模间的矛盾问题，其涉及到网络容量的可能性与可行性的关系问题，即学习复杂性问题。
* BP神经网络样本依赖性问题：网络模型的逼近和推广能力与学习样本的典型性密切相关，而从问题中选取典型样本实例组成训练集是一个很困难的问题。

#### C4.5
>优点：
* 便于理解和解释。树的结构可视化。
* 训练需要的数据少。
* 能够处理数值型数据和分类数据，其他的技术通常只能用来专门分析某一种的变量类型的数据集；
* 能够处理多路输出问题。
* 可以通过数值统计测试来验证该模型。
>缺点：
* 决策树模型容易产生一个过于复杂的模型，这样的模型对数据的泛化性能会很差，即过拟合。尤其当训练数据特征较多时，生成树结果复杂，如optdigits数据集具有64个特征。
* 决策树可能是不稳定的，因为在数据中的微小变化可能会导致完全不同的树生成。
* 在多方面性能最优和简单化概念的要求下，学习一颗最优决策树通常是一个NP难问题。
