import matplotlib as mat
mat.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#定义简单线性回归类
class linearClarfy:
    #属性个数
    n=0
    #权重
    weight=np.zeros(n)
    #中心坐标
    pos_center=np.zeros(n)
    #阀值
    T=0
    # 训练函数
    def fit(self, xtrain, ytrain):
        # 得到正例索引
        index1 = np.where(ytrain == 1)
        # 正例中心
        pos_centriod = np.mean(xtrain[index1[0]], axis=0)
        # 得到负例索引
        index2 = np.where(ytrain == 0)
        # 负例中心
        neg_centriod = np.mean(xtrain[index2[0]], axis=0)
        # 得到权重向量
        self.weight = pos_centriod - neg_centriod
        #计算中心坐标
        self.pos_center=1 / 2 * (pos_centriod + neg_centriod)
        # 计算阈值t
        self.T = np.dot(self.weight, self.pos_center)
        return self.T
    # 准确率测试
    def score(self, xtest, ytest, threshold):
        # 测试集预测类别
        predict = []
        for i in xtest:
            # 大于阈值标签为1
            if np.dot(i, self.weight) >= threshold:
                predict.append(1)
            # 反之为0
            else:
                predict.append(0)
        accuracy = accuracy_score(ytest, predict)
        return accuracy
    #准确率计算函数
    @staticmethod
    def accuracy_score(ytest,predict):
        #计数器
        count=0
        for i,j in ytest,predict:
            if i==j:
                count+=1
        return count/len(ytest)


#导入数据
a=linearClarfy()




