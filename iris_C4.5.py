from math import log
import operator
import pickle
import numpy as np
from sklearn import model_selection

def createDataSet():
    labels = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
    file_path = "./iris.data"
    # 该方法可将输入的字符串作为字典it 的键进行查询，输出对应的值
    def iris_type(s):
        it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
        return it[s]
    # 加载data文件，类型为浮点，分隔符为逗号，对第四列也就是data 中的鸢尾花类别这一列的字符串转换为0-2 的浮点数
    data = np.loadtxt(file_path, dtype=float, delimiter=',', converters={4: iris_type})
    # 对data 矩阵进行分割，从第四列包括第四列开始后续所有列进行拆分
    x, y = np.split(data, (4,), axis=1)
    # 随机分配训练数据和测试数据，随机数种子为1，测试数据占比为0.3
    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3)
    X_train = np.hstack((X_train, y_train.reshape(-1, 1)))
    X_train = X_train.tolist()
    X_test=X_test.tolist()
    y_list = []
    [y_list.append(y_test[i][0]) for i in range(len(y_test))]
    y_test = y_list
    return X_train, labels, X_test, y_test

def calcShannonEnt(dataSet):
    # 返回数据集的行数
    numEntires = len(dataSet)
    # 保存每个标签（Label）出现次数的“字典”
    labelCounts = {}
    # 对每组特征向量进行统计
    for featVec in dataSet:
        # 提取标签（Label）信息
        currentLabel = featVec[-1]
        # 如果标签（Label）没有放入统计次数的字典，添加进去
        if currentLabel not in labelCounts.keys():
            # 创建一个新的键值对，键为currentLabel值为0
            labelCounts[currentLabel] = 0
        # Label计数
        labelCounts[currentLabel] += 1
    # 经验熵（香农熵）
    shannonEnt = 0.0
    # 计算香农熵
    for key in labelCounts:
        # 选择该标签（Label）的概率
        prob = float(labelCounts[key]) / numEntires
        # 利用公式计算
        shannonEnt -= prob * log(prob, 2)
    # 返回经验熵（香农熵）
    return shannonEnt


"""
函数说明：按照给定特征划分数据集
Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    values - 需要返回的特征的值
Returns:
    None
"""


def splitDataSet(dataSet, axis, value):
    # 创建返回的数据集列表
    retDataSet = []
    # 遍历数据集的每一行
    for featVec in dataSet:
        if featVec[axis] == value:
            # 去掉axis特征
            reducedFeatVec = featVec[:axis]
            # 将符合条件的添加到返回的数据集
            # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            reducedFeatVec.extend(featVec[axis + 1:])
            # 列表中嵌套列表
            retDataSet.append(reducedFeatVec)
            # 返回划分后的数据集
    return retDataSet


def chooseBestFeatureToSplitRatio(dataSet):
    # 特征数量
    numFeatures = len(dataSet[0]) - 1  # 最后一列为label
    # 计算数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益
    bestInfoGain = 0.0
    # 最优特征的索引值
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征存在featList这个列表中（列表生成式）
        featList = [example[i] for example in dataSet]
        # 创建set集合{}，元素不可重复，重复的元素均被删掉
        # 从列表中创建集合是python语言得到列表中唯一元素值得最快方法
        uniqueVals = set(featList)
        # 经验条件熵
        newEntropy = 0.0

        IV = 0.0
        # 计算信息增益
        for value in uniqueVals:
            # subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
            IV -= prob * log(prob, 2)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        # 信息增益率
        if IV == 0:
            IV = 1
        infoGainRatio = infoGain / IV
        # 打印每个特征的信息增益
        # print("第%d个特征的增益率为%.3f" % (i, infoGainRatio))
        # 计算信息增益
        if (infoGainRatio > bestInfoGain):
            # 更新信息增益，找到最大的信息增益
            bestInfoGain = infoGainRatio
            # 记录信息增益最大的特征的索引值
            bestFeature = i
    # 返回信息增益最大的特征的索引值
    return bestFeature


"""
函数说明：统计classList中出现次数最多的元素（类标签）
        服务于递归第两个终止条件
Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现次数最多的元素（类标签）
"""


def majorityCnt(classList):
    classCount = {}
    # 统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 根据字典的值降序排序
    # operator.itemgetter(1)获取对象的第1列的值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回classList中出现次数最多的元素
    return sortedClassCount[0][0]


"""
函数说明：创建决策树
        递归有两个终止条件：1、所有的类标签完全相同，直接返回类标签
                        2、用完所有标签但是得不到唯一类别的分组，即特征不够用，挑选出现数量最多的类别作为返回
Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树
"""


def createTree(dataSet, labels):
    # 取分类标签（是否放贷：yes or no）
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的类标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优特征
    bestFeat = chooseBestFeatureToSplitRatio(dataSet)
    # 最优特征的标签
    bestFeatLabel = labels[bestFeat]
    # 根据最优特征的标签生成树
    myTree = {bestFeatLabel: {}}
    # 删除已经使用的特征标签
    del (labels[bestFeat])
    # 得到训练集中所有最优解特征的属性值
    featValues = [example[bestFeat] for example in dataSet]
    # 去掉重复的属性值
    uniqueVals = set(featValues)
    # 遍历特征，创建决策树
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

"""
函数说明：使用决策树分类
Parameters:
    inputTree - 已经生成的决策树
    featLabels - 存储选择的最优特征标签
    testVec - 测试数据列表，顺序对应最优特征标签
Returns:
    classLabel - 分类结果
"""
# 运用决策树进行分类
def classify(inputTrees, featLabels, testVec):
    firstStr = list(inputTrees.keys())[0]
    secondDict = inputTrees[firstStr]
    featIndex = featLabels.index(firstStr)  # 寻找决策属性在输入向量中的位置
    classLabel = -1  # -1是作为flag值
    for key in secondDict.keys():
        if testVec[featIndex] == key:  # 如果对应位置的值与键值相等
            if type(secondDict[key]).__name__ == 'dict':
                # 继续递归查找
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]  # 查找到子节点则返回子节点的标签
    # 标记classLabel为-1当循环过后若仍然为-1，表示未找到该数据对应的节点则我们返回他兄弟节点出现次数最多的类别
    return getLeafBestCls(inputTrees) if classLabel == -1 else classLabel


# 求该节点下所有叶子节点的列表
def getLeafscls(myTree, clsList):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            clsList = getLeafscls(secondDict[key], clsList)
        else:
            clsList.append(secondDict[key])
    return clsList


# 返回出现次数最多的类别
def getLeafBestCls(myTree):
    clsList = []
    resultList = getLeafscls(myTree, clsList)
    return max(resultList, key=resultList.count)

"""
函数说明：main函数
Parameters:
    None
Returns:
    None
"""


def main():
    dataSet, features, X_test, y_test = createDataSet()

    myTree = createTree(dataSet, features)
    featlabels = ['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']
    result = []
    for i in range(len(X_test)):
        resu = classify(myTree, featlabels, X_test[i])

        result.append(resu)
    result = np.array(result)
    y_test = np.array(y_test)
    accuracy = np.mean(result == y_test)

    print(myTree)
    print("最优特征索引值:" + str(chooseBestFeatureToSplitRatio(dataSet)))
    print('准确率:', accuracy)


if __name__ == '__main__':
    main()

