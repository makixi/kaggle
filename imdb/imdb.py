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

#��ԭʼ���ݵ�ÿһ�а�'\t'���зָ�,Ĭ����','
train=pd.read_csv('F:/demo/kaggle/imdb/labeledTrainData.tsv',delimiter='\t')
test=pd.read_csv('F:/demo/kaggle/imdb/testData.tsv',delimiter='\t')

#����Ԥ����
def review_to_text(review,remove_stopwords):
    raw_text=BeautifulSoup(review,'html').get_text() #ȥ��html���
    letters=re.sub('[^a-zA-Z]',' ',raw_text)#ֻ������ĸ�ַ�
    words=letters.lower().split()#�Ƚ�����ת��Сд��ĸ��ʾ���ٰ��տո񻮷�Ϊ����list
    if remove_stopwords:
        stop_words=set(stopwords.words('english'))
        words=[w for w in words if w not in stop_words]#ȥ��ͣ�ô�
    return words

#ԭʼ���ݺͲ������ݽ���Ԥ����
X_train=[]
for review in train['review']:
    X_train.append(' '.join(review_to_text(review,True)))

X_test=[]
for review in test['review']:
    X_test.append(' '.join(review_to_text(review,True)))

y_train=train['sentiment']
#
# #��ȡ�ı�����
# pip_count=Pipeline([('count_vec',CountVectorizer(analyzer='word')),('mnb',MultinomialNB())])
# pip_tfidf=Pipeline([('tfidf_vec',TfidfVectorizer(analyzer='word')),('mnb',MultinomialNB())])
#
# params_count={'count_vec__binary':[True,False],'count_vec__ngram_range':[(1,1),(1,2)],'mnb__alpha':[0.1,1.0,10.0]}
# params_tfidf={'tfidf_vec__binary':[True,False],'tfidf_vec__ngram_range':[(1,1),(1,2)],'mnb__alpha':[0.1,1.0,10.0]}
#
# #���л�����������
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

######δ�������
unlabeled_train=pd.read_csv('F:/demo/kaggle/imdb/unlabeledTrainData.tsv',delimiter='\t',quoting=3)

tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')

#�־�
def review_to_sentences(review,tokenizer):
    ##Python strip() ���������Ƴ��ַ���ͷβָ�����ַ���Ĭ��Ϊ�ո�
    raw_sentences=tokenizer.tokenize(review.strip())
    sentences=[]
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_to_text(raw_sentence,False))
    return sentences

#corpora(ȫ��)������洢���о���Ԥ�����ľ��ӵļ���
corpora=[]

for review in unlabeled_train['review']:
    corpora += review_to_sentences(review.encode('utf-8').decode('utf-8'),tokenizer)

#����ѵ��������ģ�͵ĳ�����
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
#���������ģ�͵�ѵ�����
model.save(model_name)


model=Word2Vec.load(model_name)

print(model.most_similar("man"))

#ʹ�ô����������ı���������
#����˼·���ǽ�һ�������������ڡ��ʻ���еĵ��ʣ�����Ӧ�Ĵ������ۼ�������
#�ٳ��Խ����˴�����ת�������е��ʵĸ���
#����Ĵʻ������ʹ��unlabeledData��ͨ��word2vec�������Ĵ�����ģ���У����ɵĴʻ��
#����������վ��ǽ�һ������ת��������������ʽ
def makeFeatureVec(words,model,num_features):
    # ��ʼ��һ��300ά������Ϊfloat32��Ԫ��ֵȫΪ0��������
    featureVec=np.zeros((num_features,),dtype="float32")
    nwords=0.
    #��ȡ�ʻ��
    index2word_set=set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords=nwords+1
            featureVec=np.add(featureVec,model[word])
    featureVec=np.divide(featureVec,nwords)
    return featureVec

#ƽ��������
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