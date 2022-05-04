import numpy as np
from sklearn import svm

#=============== 数据准备 ===============
file_path = "./optdigits.tra"
tfile_path='./optdigits.tes'

# 加载data文件，类型为浮点，分隔符为逗号
data = np.loadtxt(file_path, dtype=float, delimiter=',')
test_data = np.loadtxt(tfile_path, dtype=float, delimiter=',')
# print(data)

# 对data 矩阵进行分割，从第64列包括第64列开始后续所有列进行拆分
x, y = np.split(data, (64,), axis=1)
tx,ty=np.split(test_data,(64,),axis=1)
# 对x 矩阵进行切片，所有行都取
x = x[:, 0:60]
tx=tx[:,0:60]
# print(x)

data_train=x
data_test=tx
tag_train=y
tag_test=ty

#=============== 模型搭建 ===============
def classifier():
    clf = svm.SVC(C=0.5,                            # 误差惩罚系数，默认1
                  kernel='rbf',                  # 线性核 kenrel='rbf':高斯核
                  decision_function_shape='ovr')    # 决策函数
    return clf

# 定义SVM（支持向量机）模型
clf = classifier()

#=============== 模型训练 ===============
def train(clf, x_train, y_train):
    clf.fit(x_train,            # 训练集特征向量
            y_train.ravel())    # 训练集目标值

# 训练SVM 模型
train(clf, data_train, tag_train)

#=============== 模型评估 ===============
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s Accuracy:%.3f' % (tip, np.mean(acc)))

def print_accuracy(clf, x_train, y_train, x_test, y_test):
    # 分别打印训练集和测试集的准确率
    # score(x_train, y_train):表示输出x_train, y_train 在模型上的准确率
    print('training prediction:%.3f' % (clf.score(x_train, y_train)))
    print('test data prediction:%.3f' % (clf.score(x_test, y_test)))
    show_accuracy(clf.predict(x_train), y_train, 'training data')
    show_accuracy(clf.predict(x_test), y_test, 'testing data')


print_accuracy(clf, data_train, tag_train, data_test, tag_test)

