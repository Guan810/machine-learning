import numpy as np
from numpy import linalg as lin

# 定义线性回归类
class linearReg:
    # 属性个数
    n=0
    # 权重
    weight=np.zeros(n)
    def fit(self, xtrain, ytrain):
        self.weight = np.dot(
            np.matmul(lin.inv(np.matmul(xtrain.T, xtrain)), xtrain.T), ytrain
        )

    # 准确率测试
    def score(self, xtest, ytest, threshold=0):
        predict = []
        for i in xtest:
            if np.dot(i, self.weight) > 0:
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
