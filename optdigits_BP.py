import numpy
# scipy.special 用于 sigmoid 函数 expit()
import scipy.special
# 用于绘制数组的库
import matplotlib.pyplot
# 确保绘图在此窗口中，而不是外部窗口
# 从 PNG 图像文件加载数据的助手
import imageio
import time
start=time. time ()
# 中间写上代码块


# 神经网络类定义
class neuralNetwork:

    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 设置每个输入、隐藏、输出层的节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 链接权重矩阵, wih and who
        # 矩阵中的权重是 w_i_j，其中链接是从节点 i 到下一层的节点 j
        # w11 w21
        # w12 w22 等
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))  # 随机取,这个是一个经验函数
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        # 从正态分布中抽取随机数据，参数为均值、标准差、维度

        # 学习率是0.1
        self.lr = learningrate

        # 激活函数是sigmoid函数
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # 训练神经网络
    def train(self, inputs_list, targets_list):
        # 将输入列表转换为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 计算信号到隐藏层
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算从隐藏层出现的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算信号到最终输出层
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算从最终输出层出现的信号
        final_outputs = self.activation_function(final_inputs)

        # 输出层误差是（目标值 - 真实值，即target - actual)
        output_errors = targets - final_outputs
        # 隐藏层误差是 output_errors，按权重分割，在隐藏节点处重新组合
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 更新隐藏层和输出层之间链接的权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # 更新输入层和隐藏层之间链接的权重
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # 查询神经网络
    def query(self, inputs_list):
        # 将输入列表转换为二维数组，T指矩阵转置,ndmin= 定义数组的最小维度,将行矩阵转为列矩阵
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 计算信号到隐藏层，矩阵点积计算隐藏层的输入值，每个输入乘上权重累加
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算从隐藏层出现的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算信号到最终输出层
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算从最终输出层出现的信号，经过每层都要用激活函数
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# 输入、隐藏和输出节点的数量
input_nodes = 64
hidden_nodes = 60
output_nodes = 10

# 学习率
learning_rate = 0.1

# 创建神经网络实例
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open(r'./optdigits.tra')
training_data_list = training_data_file.readlines()
# 每行都是一个输入数据
training_data_file.close()

# 训练神经网络
# epochs 是训练数据集用于训练的次数，即迭代次数
epochs = 10  # 设置迭代次数

for e in range(epochs):
    # 遍历训练数据集中的所有‘记录’
    for record in training_data_list:
        # 用 ',' 逗号分割记录
        all_values = record.split(',')
        # 缩放和转化输入，asfarray将列表转化为数组
        inputs = (numpy.asfarray(all_values[:64]) / 255.0 * 0.99) + 0.01
        # 创建目标输出值（全部为 0.01，除了所需的标签为 0.99）
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] 是这条记录的目标标签
        targets[int(all_values[64])] = 0.99
        # int(all_values[0])表示输入的数字，及识别的目标
        n.train(inputs, targets)

#模型评估
f=open(r'./optdigits.tes')
data=f.readlines()
f.close()

real=[]
for i in data:
   real.append(i[-2])
y=[]
for i in data:
   i=i.split(',')
   outputs=n.query(numpy.asfarray(i[:64]))
   y.append(numpy.argmax(outputs))

arr2 = [int(x) for x in real]
from sklearn.metrics import classification_report
print(classification_report(arr2,y))
#macro avg = 上面类别各分数的直接平均
#weighted avg=上面类别各分数的加权（权值为support）平均
#support意思为支持，也就是说真实结果中有多少是该类别。
end=time. time ()
print ( 'Running time: %s Seconds' %(end-start))