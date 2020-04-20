import matplotlib as mat
mat.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
if __name__ == "__main__":
    data=load_iris()  # 导入数据
    radio=7/3  # 划分比例
    x=data.get("data") #获取数据矩阵
    label=data.get("target") #获取标签向量
    num=x.shape[0] #样本数量
    num_test=num/(1-radio)
    num_tarin=num-num_test
    index=np.arange(num) #产生样本编号
    np.random.shuffle(index)
    x_test=x[index[:num_test]]
    y_test=y[index[:num_test]]
    

