#调用sklearn库的svc方法实例化模型对象
#调用sklearn.model_selection库的train_test_split方法划分训练集和测试集
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#导入经word2vec提取的特征
index="5"
with open('result/'+index+'/word2vec_feature.pkl','rb') as file:
    tfidf_feature=pickle.load(file)
    X=tfidf_feature['feartureMatrix']
    y=tfidf_feature['label']

#划分数据集
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=1)
clf=svm.SVC(C=8,kernel='rbf',gamma=0.06,decision_function_shape="ovo")
clf.fit(train_X,train_y)
with open("result/"+index+"/result.txt","w") as file:
    file.write(clf.score(test_X,test_y))
    file.write("/n")
    #生成测试报告
    y_pred=clf.predict(test_X)
    file.write(classification_report(y_true=test_y,y_pred=y_pred))

