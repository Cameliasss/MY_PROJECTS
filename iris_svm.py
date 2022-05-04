import numpy as np
from sklearn import svm
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl

#=============== 数据准备 ===============
file_path = "./iris.data"

# 该方法可将输入的字符串作为字典it 的键进行查询，输出对应的值
# 该方法就是相当于一个转换器，将数据中非浮点类型的字符串转化为浮点
def iris_type(s):
    it = {b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2}
    return it[s]

# 加载data文件，类型为浮点，分隔符为逗号，对第四列也就是data 中的鸢尾花类别这一列的字符串转换为0-2 的浮点数
data = np.loadtxt(file_path, dtype=float, delimiter=',', converters={4:iris_type})
# print(data)

# 对data 矩阵进行分割，从第四列包括第四列开始后续所有列进行拆分
x, y = np.split(data, (4,), axis=1)
# 对x 矩阵进行切片，所有行都取，但只取前两列
x = x[:, 2:4]
# print(x)

# 随机分配训练数据和测试数据，随机数种子为1，测试数据占比为0.3
data_train, data_test, tag_train, tag_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3)

#=============== 模型搭建 ===============
def classifier():
    clf = svm.SVC(C=0.5,                            # 误差惩罚系数，默认1
                  kernel='linear',                  # 线性核 kenrel='rbf':高斯核
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
    # 原始结果与预测结果进行对比
    # predict() 表示对x_train 样本进行预测，返回样本类别
    show_accuracy(clf.predict(x_train), y_train, 'training data')
    show_accuracy(clf.predict(x_test), y_test, 'testing data')
    # 计算决策函数的值，表示x到各分割平面的距离
    # print('decision_function:\n', clf.decision_function(x_train))

print_accuracy(clf, data_train, tag_train, data_test, tag_test)

#=============== 模型可视化 ===============
def draw(clf, x):
    iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'
    # 开始画图
    # 第0 列的范围
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    # 第1 列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    # print('grid_test:\n', grid_test)
    # 输出样本到决策面的距离
    z = clf.decision_function(grid_test)
    # print('the distance to decision plane:\n', z)
    # 预测分类值
    grid_hat = clf.predict(grid_test)
    # print('grid_hat:\n', grid_hat)
    grid_hat = grid_hat.reshape(x1.shape)

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    # 样本点
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)
    # 测试点
    plt.scatter(data_test[:, 0], data_test[:, 1], s=120, facecolor='none', zorder=10)
    plt.xlabel(iris_feature[0], fontsize=20)
    plt.ylabel(iris_feature[1], fontsize=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('svm in iris data classification', fontsize=30)
    plt.grid()
    plt.show()

draw(clf, x)


