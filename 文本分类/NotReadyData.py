import pandas as pd
import numpy as np
import smart_open
from gensim.models import Word2Vec
import jieba
import pickle
from sklearn.preprocessing import LabelEncoder
#加载数据
data_df=pd.read_csv('data.txt',sep='\t',header=None)
data_df.columns=['分类','内容']

#读取停顿词列表
stopword_list=[k.strip() for k in open('hit_stopwords.txt',encoding='utf8').readlines() if k.strip()!=""]
stopword_list.append(" ")
#获得分词列表,判断是否为停顿词
cutwords_list=[]
index="5"
for article in data_df['内容']:
    cutwords=[k for k in jieba.cut(article) if k not in stopword_list]
    cutwords_list.append(cutwords)
    #将分词结果保存为本地文件
with open("result/"+index+"/cutwords_list.txt",'w') as file:
    for cutwords in cutwords_list:
        file.write(' '.join(cutwords)+'\n')
with open('result/'+index+'/cutwords_list.txt') as file:
    cutwords_list=[k.split() for k in file.readlines()]
#对于每一个标题,获取每一个分词在word2vec模型下的相关性向量,把这个标题的所有分词在word2vec模型的相关性向量求和取平均数,作为此标题在word2vec模型的相关性结果
def getVector(cutwords,word2vec_model):
    count=0
    article_vector=np.zeros(word2vec_model.layer1_size)
    for cutword in cutwords:
        if cutword in word2vec_model:
            article_vector +=word2vec_model[cutword]
            count +=1
    if count>0:
        return article_vector/count
    else:
        return article_vector
vector_list=[]
model = Word2Vec(cutwords_list, sg=1, size=100,  window=5,  min_count=int(index),  negative=3, sample=0.001, hs=1)
for cutwords in cutwords_list:
    vector_list.append(getVector(cutwords,model))
X=np.array(vector_list)
labelEncoder=LabelEncoder()
y=labelEncoder.fit_transform(data_df['分类'])
with open('result/'+index+'/word2vec_feature.pkl','wb') as file:
    save={
        'feartureMatrix':X,
        'label':y
    }
    pickle.dump(save,file)

