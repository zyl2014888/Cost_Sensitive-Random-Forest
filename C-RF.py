# coding:utf-8
"""
作者：ZYL	日期：2019.10.20
功能：代价敏感随机森林，Cost-Sensitive Random Forest（C-RF），wine数据集[1,2]二分类
版本：1.0
"""
from __future__ import division
import pandas as pd
import copy
import random
import math

#用于统计训练阶段的FN，FP数量的类
class treeMsg():
    def __init__(self,FN,FP):
        self.FN = FN
        self.FP = FP



# 最后一个属性还不能将样本完全分开，此时数量最多的label被选为最终类别   'ok'
def majorClass(classList):
    classDict = {}
    for cls in classList:
        classDict[cls] = classDict.get(cls, 0) + 1
    sortClass = sorted(classDict.items(), key=lambda item: item[1])
    return sortClass[-1][0]



# 计算基尼系数    'ok'
def calcGini(dataSet):
    labelCounts = {}
    # 给所有可能分类创建字典
    for dt in dataSet:
        currentLabel = dt[-1]#最后一个数据（标签）
        #统计正负例的个数
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1 #查找字典中currentLabel，如果不在，则返回0
    Gini = 1
    for key in labelCounts:
        #有正例计算整理的比率，没有就计算负例的比例
        prob = labelCounts[key] / len(dataSet)      #key=标签
        a=(1-prob)**2
        Gini -= prob * prob     #Gini系数就是 1-方差
    return Gini

# 对连续变量划分数据集
def splitDataSet(dataSet, featIndex, value):
    leftData, rightData = [], []
    for dt in dataSet:
        if dt[featIndex] <= value:
            leftData.append(dt)
        else:
            rightData.append(dt)
    return leftData, rightData

# 选择最好的数据集划分方式   'ok'
def chooseBestFeature(dataSet):
    bestGini = 1
    bestFeatureIndex = -1
    bestSplitValue = None
    # 第i个特征
    for i in range(len(dataSet[0]) - 1):
        featList = [dt[i] for dt in dataSet]
        # 产生候选划分点
        sortfeatList = sorted(list(set(featList)))
        splitList = []
        for j in range(len(sortfeatList) - 1):
            splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2)

        # 第j个候选划分点，记录最佳划分点
        for splitValue in splitList:
            newGini = 0
            subDataSet0, subDataSet1 = splitDataSet(dataSet, i, splitValue)
            newGini += len(subDataSet0) / len(dataSet) * calcGini(subDataSet0)
            newGini += len(subDataSet1) / len(dataSet) * calcGini(subDataSet1)
            if newGini < bestGini:
                bestGini = newGini
                bestFeatureIndex = i
                bestSplitValue = splitValue
    return bestFeatureIndex, bestSplitValue

# 去掉第i个属性，生成新的数据集
def splitData(dataSet, featIndex, features, value):
    newFeatures = copy.deepcopy(features)
    newFeatures.remove(features[featIndex])
    leftData, rightData = [], []
    for dt in dataSet:
        temp = []
        temp.extend(dt[:featIndex])
        temp.extend(dt[featIndex + 1:])
        if dt[featIndex] <= float(value):
            leftData.append(temp)
        else:
            rightData.append(temp)

    return newFeatures, leftData, rightData

# 建立决策树   'ok'
def createTree(dataSet, features):
    classList = [dt[-1] for dt in dataSet]#提取标签
    # label一样，全部分到一边
    if classList.count(classList[0]) == len(classList):
        #print(len(classList),classList[0],classList.count(classList[0]))
        #len(classList):样本数量    classList[0]:第一个标签   classList.count(classList[0])：第一个标签的数量
        return classList[0]
    # 最后一个特征还不能把所有样本分到一边，则选数量最多的label
    if len(features) == 1:#当特征是最后一个时，进入majorClass函数
        return majorClass(classList)
    bestFeatureIndex, bestSplitValue = chooseBestFeature(dataSet)
    bestFeature = features[bestFeatureIndex]
    # 生成新的去掉bestFeature特征的数据集
    newFeatures, leftData, rightData = splitData(dataSet, bestFeatureIndex, features, bestSplitValue)
    # 左右两颗子树，左边小于等于最佳划分点，右边大于最佳划分点
    myTree = {bestFeature: {'<' + str(bestSplitValue): {}, '>' + str(bestSplitValue): {}}}
    myTree[bestFeature]['<' + str(bestSplitValue)] = createTree(leftData, newFeatures)
    myTree[bestFeature]['>' + str(bestSplitValue)] = createTree(rightData, newFeatures)
    return myTree

# 用生成的决策树对测试样本进行分类
def treeClassify(decisionTree, featureLabel, testDataSet):
    firstFeature = list(decisionTree.keys())[0]
    secondFeatDict = decisionTree[firstFeature]
    splitValue = float(list(secondFeatDict.keys())[0][1:])
    featureIndex = featureLabel.index(firstFeature)

    if testDataSet[featureIndex] <= splitValue:
        valueOfFeat = secondFeatDict['<' + str(splitValue)]
    else:
        valueOfFeat = secondFeatDict['>' + str(splitValue)]
    if isinstance(valueOfFeat, dict):
        pred_label = treeClassify(valueOfFeat, featureLabel, testDataSet)
    else:
        pred_label = valueOfFeat
    return pred_label

# 随机抽取样本，样本数量与原训练样本集一样，维度为sqrt(m-1)   'ok'
def baggingDataSet(dataSet):
    n, m = dataSet.shape#n=数据个数  m=特征数量

    features = random.sample(list(dataSet.columns.values[:-1]), int(math.sqrt(m - 1)))#从list随机选取多少个元素
    features.append(dataSet.columns.values[-1])#将标签加入到features中，上一步取特特征不包括标签
    rows = [random.randint(0, n-1) for _ in range(n)]#在[0，n-1]中生成随机数  for _ in range(n)  _ 是占位符，只代表循环几次，不管什么值
    trainData = dataSet.iloc[rows][features]#取矩阵中第row行第features列的那个元素
    return trainData.values.tolist(), features





def testWine():
    df = pd.read_csv('wine.txt', header=None)
    labeList = []  # 预测标签
    dataList = []  # 数据本身标签
    data=[] #训练数据
    Data_T=0
    Data_F=0
    for i in open('wine.txt','r').readlines():
        l=i.split(',')
        dataList.append(int(i.split(',')[-1]))
        if int(i.split(',')[-1])==1:
            Data_T+=1
        else:
            Data_F+=1

    for i in open('wine.txt','r').readlines():
        l=i.split(',')
        t=[]
        for j in range(len(l)-1):
            t.append(float(l[j]))
        data.append(t)




    Test_T=0
    Test_F=0
    file = 'winet.txt'
    f = open(file, 'r')
    Test = []
    TestLabel = []
    labelList = []
    for i in f.readlines():
        l = i.split(',')
        if int(l[-1])==1:
            Test_T+=1
        else:
            Test_F+=1
        TestLabel.append(int(l[-1]))
        t=[]
        for j in range(len(l)-1):
            t.append(float(l[j]))
        Test.append(t)
    labels = df.columns.values.tolist()#取列表名字
    #df = df[df[labels[-1]] != 3]#列表反向取！=3的第一个元素,为了变成二分类问题
    # 生成多棵决策树，放到一个list里边


    treeCounts = 110#决策树数量
    #while 1:
    treeList = []
    for i in range(treeCounts):
        baggingData, bagginglabels = baggingDataSet(df)
        decisionTree = createTree(baggingData, bagginglabels)
        treeList.append(decisionTree)
    # 对测试样本分类
    labelPred = []
    T_AC = [] #统计每棵树经过训练样本后的FN和FP
    AC=[]#计算每棵树的平均误差代价   AC的第一个元素是AC值，第二个元素是该AC值对应的树的位置
    for i in range(len(data)):
        testData=data[i]
        for j in range(len(treeList)):
            if i==0:
                t=treeMsg(0,0)
                T_AC.append(t)#放每个树的AC类
            label = treeClassify(treeList[j], labels[:-1], testData)
            if int(label)!=dataList[i]:
                if dataList[i]==2:
                    T_AC[j].FP+=1
                else:
                    T_AC[j].FN+=1
            labelPred.append(label)
        # 投票选择最终类别
        labelDict = {}
        for label in labelPred:
            labelDict[label] = labelDict.get(label, 0) + 1
        sortClass = sorted(labelDict.items(), key=lambda item: item[1])
        #print("The predicted label is: {}".format(sortClass[-1][0])+'  '+str(i))
        labeList.append(int(sortClass[-1][0]))
    acc=CluAcc(labeList, dataList)
    print('训练量:{Data}  测试量:{Test}  训练正负分布: {D_T}:{D_F}  测试正负分布: {T_T}:{T_F}  准确率:{acc}'.format(Data=df.shape[0],Test=len(Test),D_T=Data_T,D_F=Data_F,T_T=Test_T,T_F=Test_F,acc=acc[0]))
    labeList=[]

    #计算每棵树的AC值
    AC=CluACTree(treeList,AC,T_AC,dataList)


    #看看对于重新训练数据的重新组合后的训练结果
    print('--------------引入CS的RF-------------')
    Cost_Pre(AC,treeList,labels,data,dataList)


    #未引入CS的RF
    Test_Pre(treeList,labels,Test,TestLabel)

    #测试数据结果
    Cost_Pre(AC,treeList,labels,Test,TestLabel)

def CluACTree(treeList,AC,T_AC,dataList):
    # 计算每棵树的AC值
    for i in range(len(treeList)):
        t = []
        t.append(float((T_AC[i].FP*1 + T_AC[i].FN*16) / len(dataList)))
        t.append(i)
        AC.append(t)
    AC.sort()  # 排序，选择最小AC值的前100棵树(总共训练了200棵树),重新选取随机森林决策树的组成
    return AC



#AC:包含每棵树的AC值和该树的位置
#tree：决策树的列表
#labels:特征的数量
#DataSet：测试数据
#labList：测试数据的本身标签
def Cost_Pre(AC,tree,labels,DataSet,labList):
    labelPred=[]
    labelist=[]
    for i in range(len(DataSet)):
        testData=DataSet[i]
        for j in range(0,100):
            label=treeClassify(tree[AC[j][1]],labels[:-1],testData)
            labelPred.append(label)
        labelDict = {}
        for label in labelPred:
            labelDict[label] = labelDict.get(label, 0) + 1
        sortClass = sorted(labelDict.items(), key=lambda item: item[1])
        labelist.append(int(sortClass[-1][0]))
    acc=CluAcc(labelist,labList)#前边是预测的标签，后边是本身的标签
    print('改进后的准确率: {:.4f}'.format(acc[0]))
    print('测试数据总共；%d' % len(DataSet))
    print('测试数据预测分类标签列表：', labelist)

def Test_Pre(tree,labels,DataSet,labList):
    print('----------------未引入CS的RF测试数据----------------')
    labelPred = []
    labelist = []
    for i in range(len(DataSet)):
        testData = DataSet[i]
        for j in range(len(tree)):
            label = treeClassify(tree[j], labels[:-1], testData)
            labelPred.append(label)
        labelDict = {}
        for label in labelPred:
            labelDict[label] = labelDict.get(label, 0) + 1
        sortClass = sorted(labelDict.items(), key=lambda item: item[1])
        labelist.append(int(sortClass[-1][0]))
    acc = CluAcc(labelist, labList)  # 前边是预测的标签，后边是本身的标签
    print('改进前的准确率: {:.4f}'.format(acc[0]))


#计算代价矩阵相关数据
#FN:假负例数
#FP:假正例数
#TC:分类时产生的总错误代价
#AC:平均误差代价




#计算正确率
def CluAcc(labelList,dataLabel):
    acc=0
    FN=0
    FP=0
    for i in range(len(labelList)):
        if labelList[i]==dataLabel[i]:
            acc+=1
        else:
            if labelList[i]==1:
                FP+=1
            else:
                FN+=1
    print('FP:%d'%FP,'FN:%d'%FN)
    return float(acc/len(dataLabel)),FP,FN

testWine()
