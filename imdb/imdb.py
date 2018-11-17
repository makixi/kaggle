import pandas as pd
import numpy as np
import re
import nltk.data
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from gensim.models import word2vec,Word2Vec

from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

#把原始数据的每一行按'\t'进行分割,默认是','
train=pd.read_csv('F:/demo/kaggle/imdb/labeledTrainData.tsv',delimiter='\t')
test=pd.read_csv('F:/demo/kaggle/imdb/testData.tsv',delimiter='\t')

#数据预处理
def review_to_text(review,remove_stopwords):
    raw_text=BeautifulSoup(review,'html').get_text() #去掉html标记
    letters=re.sub('[^a-zA-Z]',' ',raw_text)#只保留字母字符
    words=letters.lower().split()#先将句子转成小写字母表示，再按照空格划分为单词list
    if remove_stopwords:
        stop_words=set(stopwords.words('english'))
        words=[w for w in words if w not in stop_words]#去掉停用词
    return words

#原始数据和测试数据进行预处理
X_train=[]
for review in train['review']:
    X_train.append(' '.join(review_to_text(review,True)))

X_test=[]
for review in test['review']:
    X_test.append(' '.join(review_to_text(review,True)))

y_train=train['sentiment']
#
# #抽取文本特征
# pip_count=Pipeline([('count_vec',CountVectorizer(analyzer='word')),('mnb',MultinomialNB())])
# pip_tfidf=Pipeline([('tfidf_vec',TfidfVectorizer(analyzer='word')),('mnb',MultinomialNB())])
#
# params_count={'count_vec__binary':[True,False],'count_vec__ngram_range':[(1,1),(1,2)],'mnb__alpha':[0.1,1.0,10.0]}
# params_tfidf={'tfidf_vec__binary':[True,False],'tfidf_vec__ngram_range':[(1,1),(1,2)],'mnb__alpha':[0.1,1.0,10.0]}
#
# #并行化超参数搜索
# gs_count=GridSearchCV(pip_count,params_count,cv=4,n_jobs=-1,verbose=1)
# gs_count.fit(X_train,y_train)
#
# print(gs_count.best_score_)
# print(gs_count.best_params_)
#
# count_y_predict=gs_count.predict(X_test)
#
#
# gs_tfidf=GridSearchCV(pip_tfidf,params_tfidf,cv=4,n_jobs=-1,verbose=1)
# gs_tfidf.fit(X_train,y_train)
#
# print(gs_tfidf.best_score_)
# print(gs_tfidf.best_params_)
#
# tfidf_y_predict=gs_tfidf.predict(X_test)
#
# submission_count=pd.DataFrame({'id':test['id'],'sentiment':count_y_predict})
# submission_tfidf=pd.DataFrame({'id':test['id'],'sentiment':tfidf_y_predict})
#
# submission_count.to_csv('F:/demo/kaggle/imdb/submission_count.csv',index=False)
# submission_tfidf.to_csv('F:/demo/kaggle/imdb/submission_tfidf.csv',index=False)

######未标记数据
unlabeled_train=pd.read_csv('F:/demo/kaggle/imdb/unlabeledTrainData.tsv',delimiter='\t',quoting=3)

tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')

#分句
def review_to_sentences(review,tokenizer):
    ##Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    raw_sentences=tokenizer.tokenize(review.strip())
    sentences=[]
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_to_text(raw_sentence,False))
    return sentences

#corpora(全集)，这里存储所有经过预处理后的句子的集合
corpora=[]

for review in unlabeled_train['review']:
    corpora += review_to_sentences(review.encode('utf-8').decode('utf-8'),tokenizer)

#配置训练词向量模型的超参数
num_features=300
min_word_count=20
num_workers=4
context=10
downsampling=1e-3

model=word2vec.Word2Vec(corpora,workers=num_workers,
        size=num_features,min_count=min_word_count,
        window=context,sample=downsampling)

model.init_sims(replace=True)

model_name='F:/demo/kaggle/imdb/300features_20minwords_10xontext'
#保存词向量模型的训练结果
model.save(model_name)


model=Word2Vec.load(model_name)

print(model.most_similar("man"))

#使用词向量产生文本特征向量
#大致思路就是将一个句子中所有在“词汇表”中的单词，所对应的词向量累加起来；
#再除以进行了词向量转换的所有单词的个数
#这里的词汇表，就是使用unlabeledData，通过word2vec所构建的词向量模型中，生成的词汇表
#这个方法最终就是将一个句子转成特征向量的形式
def makeFeatureVec(words,model,num_features):
    # 初始化一个300维，类型为float32，元素值全为0的列向量
    featureVec=np.zeros((num_features,),dtype="float32")
    nwords=0.
    #获取词汇表
    index2word_set=set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords=nwords+1
            featureVec=np.add(featureVec,model[word])
    featureVec=np.divide(featureVec,nwords)
    return featureVec

#平均词向量
def getAvgFeatureVecs(reviews,model,num_features):
    counter=0
    reviewFeatureVecs=np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter]=makeFeatureVec(review,model,num_features)
        counter+=1
    return reviewFeatureVecs

clean_train_reviews=[]
for review in train["review"]:
    clean_train_reviews.append(review_to_text(review,remove_stopwords=True))

trainDataVecs=getAvgFeatureVecs(clean_train_reviews,model,num_features)

clean_train_reviews=[]
for review in test["review"]:
    clean_train_reviews.append(review_to_text(review, remove_stopwords=True))

testDataVecs=getAvgFeatureVecs(clean_train_reviews,model,num_features)

gbc=GradientBoostingClassifier()
params_gbc={'n_estimators':[10,100,500],'learning_rate':[0.01,0.1,1.0],'max_depth':[2,3,4]}

gs=GridSearchCV(gbc,params_gbc,cv=4,n_jobs=-1,verbose=1)
gs.fit(trainDataVecs,y_train)

print(gs.best_score_)
print(gs.best_params_)

result=gs.predict(testDataVecs)
output=pd.DataFrame(data={"id":test["id"],"sentiment":result})
output.to_csv("F:/demo/kaggle/imdb/submission_w2v.csv",index=False,quoting=3)