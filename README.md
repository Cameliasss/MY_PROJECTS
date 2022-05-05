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
