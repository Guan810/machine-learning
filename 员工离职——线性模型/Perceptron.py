import numpy as np
from numpy import linalg as lin

# 定义感知器类
class perceptron:
    # 属性个数
    n=0
    # 权重
    weight=np.zeros(n)
    # 训练速率
    __learning_rate=np.zeros(n)
    def fit(self, xtrain, ytrain):
        # 找到训练集中的所有正例
        indexs = np.where(ytrain == 1)
        pos = xtrain[indexs[0], :]
        # 对于误分类的正例进行权重学习
        # 直到所有正例都被正确分类，退出迭代，否则达到最大迭代次数，退出迭代
        count = 0
        for ite in range(500):
            for i in pos:
                if np.dot(i, self.weight) < 0:
                    self.weight += self.__learning_rate * np.array(i).reshape(-1, 1)
                else:
                    # 正确分类个数加1
                    count += 1
            if count == len(indexs[0]):
                break
    # 准确率测试
    def score(self, xtest, ytest, threshold=0):
        predict = []
        for i in xtest:
            if np.dot(i, self.weight) >= 0:
                predict.append(1)
            else:
                predict.append(0)
        accuracy = accuracy_score(ytest, predict)
        return accuracy
    # 准确度计算函数
    def accuracy_score(self, ytest, predict):
        # 计数器
        count = 0
        for i, j in ytest, predict:
            if i == j:
                count += 1
        return count / len(ytest)