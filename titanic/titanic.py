import pandas as pd
import re as re
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score,ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,NuSVC,LinearSVC
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,BaggingClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,PassiveAggressiveClassifier,RidgeClassifierCV,SGDClassifier,Perceptron
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB

train=pd.read_csv('F:/demo/kaggle/titanic/train.csv')
test=pd.read_csv('F:/demo/kaggle/titanic/test.csv')

#print(train.info())
#print(train.head())

#各个class的生还率
#print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean())

full_data=[train,test]

#加入新特征
for i in full_data:
    i['FamilySize'] = i['SibSp'] + i['Parch'] + 1
# 考虑FamilySize
#print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean())

#考虑船上是否是只有一个人
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# 考虑Name，重点关注每个人的title
# 创建新特征Title
def get_title(name):
    # 正则式第一个字符为空格不要忘
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # 如果有title返回它
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

#print(pd.crosstab(train['Title'], train['Sex']))

selected_features=['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare','FamilySize','Title','IsAlone']

X_train=train[selected_features]
X_test=test[selected_features]



y_train=train['Survived'] #最后输出的结果

#可知embarked中s出现较多 na值由S填充
#print(X_train['Embarked'].value_count())
#print(X_test['Embarked'].value_count())

X_train['Embarked'].fillna('S',inplace=True)
X_test['Embarked'].fillna('S',inplace=True)

#对于age和fare的na值，用平均值填充
X_train['Age'].fillna(X_train['Age'].median(),inplace=True)
X_test['Age'].fillna(X_test['Age'].median(),inplace=True)
#X_train['Fare'].fillna(X_train['Fare'].mean(),inplace=True)
X_test['Fare'].fillna(X_test['Fare'].median(),inplace=True)

dict_vec=DictVectorizer(sparse=False)#转换为特征向量 不产生稀疏矩阵
X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))#转换成记录形式
X_test=dict_vec.transform(X_test.to_dict(orient='record'))#转换成记录形式
#print(dict_vec.feature_names_)

# Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    # Ensemble Methods
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    RandomForestClassifier(),

    # Gaussian Processes
    GaussianProcessClassifier(),

    # GLM
    LogisticRegressionCV(),
    PassiveAggressiveClassifier(),
    RidgeClassifierCV(),
    SGDClassifier(),
    Perceptron(),

    # Navies Bayes
    BernoulliNB(),
    GaussianNB(),

    # Nearest Neighbor
    KNeighborsClassifier(),

    # SVM
    SVC(probability=True),
    NuSVC(probability=True),
    LinearSVC(),

    # Trees
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),

    # Discriminant Analysis
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),

    # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()
]

vote_est = [
    # Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
    ('ada', AdaBoostClassifier()),
    ('bc', BaggingClassifier()),
    ('etc', ExtraTreesClassifier()),
    ('gbc', GradientBoostingClassifier()),
    ('rfc', RandomForestClassifier()),

    # Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
    ('gpc', GaussianProcessClassifier()),

    # GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ('lr', LogisticRegressionCV()),

    # Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
    ('bnb', BernoulliNB()),
    ('gnb', GaussianNB()),

    # Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
    ('knn', KNeighborsClassifier()),

    # SVM: http://scikit-learn.org/stable/modules/svm.html
    ('svc', SVC(probability=True)),

    # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    ('xgb', XGBClassifier())

]

grid_n_estimator = range(20,1100,20)
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [0.01,0.05,0.1,0.2,0.25,0.3,0.5,0.7,0.75,0.8,1.0]
grid_max_depth = range(2,15)
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

grid_param = [
    [{
        # AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
        'n_estimators': grid_n_estimator,  # default=50
        'learning_rate': grid_learn,  # default=1
         'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
        'random_state': grid_seed
    }],

    [{
        # BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
        'n_estimators': grid_n_estimator,  # default=10
        'max_samples': grid_ratio,  # default=1.0
        'random_state': grid_seed
    }],

    [{
        # ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
        'n_estimators': grid_n_estimator,  # default=10
        'criterion': grid_criterion,  # default=”gini”
        'max_depth': grid_max_depth,  # default=None
        'random_state': grid_seed
    }],

    [{
        # GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
         'loss': ['deviance', 'exponential'], #default=’deviance’
        'learning_rate': [.05],
    # default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
        'n_estimators': [300],
    # default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
         'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
        'max_depth': grid_max_depth,  # default=3
        'random_state': grid_seed
    }],

    [{
        # RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
        'n_estimators': grid_n_estimator,  # default=10
        'criterion': grid_criterion,  # default=”gini”
        'max_depth': grid_max_depth,  # default=None
        'oob_score': [True],
    # default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
        'random_state': grid_seed
    }],

    [{
        # GaussianProcessClassifier
        'max_iter_predict': grid_n_estimator,  # default: 100
        'random_state': grid_seed
    }],

    [{
        # LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
        'fit_intercept': grid_bool,  # default: True
         'penalty': ['l1','l2'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # default: lbfgs
        'random_state': grid_seed
    }],

    [{
        # BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
        'alpha': grid_ratio,  # default: 1.0
    }],

    # GaussianNB -
    [{}],

    [{
        # KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7],  # default: 5
        'weights': ['uniform', 'distance'],  # default = ‘uniform’
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }],

    [{
        # SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
        # http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [1, 2, 3, 4, 5],  # default=1.0
        'gamma': grid_ratio,  # edfault: auto
        'decision_function_shape': ['ovo', 'ovr'],  # default:ovr
        'probability': [True],
        'random_state': grid_seed
    }],

    [{
        # XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
        'learning_rate': grid_learn,  # default: .3
        'max_depth': range(2,15),  # default 2
        'n_estimators': grid_n_estimator,
        'seed': grid_seed
    }]
]


params={'max_depth':range(2,15),'n_estimators':range(100,1100,200),'learning_rate':[0.05,0.1,0.2,0.25,0.3,0.5,0.7,0.75,0.8,1.0]}
cv_split = ShuffleSplit(n_splits=10, test_size=.3, train_size=.6,random_state=0)
cnt=0
for clf, param in zip(vote_est, grid_param):  # https://docs.python.org/3/library/functions.html#zip
    best_search = GridSearchCV(estimator=clf[1], param_grid=param, cv=cv_split, scoring='roc_auc')
    best_search.fit(X_train, y_train)
    y_predict = best_search.predict(X_test)
    print("best"+str(clf[1]))
    print(best_search.best_score_)
    print(best_search.best_params_)
    rfc_best_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_predict})
    rfc_best_submission.to_csv('F:/demo/kaggle/titanic/'+str(cnt)+'_best_submission.csv',index=False)
    cnt+=1


for each in MLA:
    f=each
    print(str(f))
    print(cross_val_score(f,X_train,y_train,cv=5).mean())
    f.fit(X_train, y_train)
    f_y_predict = f.predict(X_test)
    f_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': f_y_predict})
    gg='F:/demo/kaggle/titanic/'+str(f)[0:5]+'_submission.csv'
    f_submission.to_csv(gg, index=False)

#
# rfc=RandomForestClassifier()
#
# xgbc=XGBClassifier()
#
# svc=SVC()
#
# #5折交叉验证
# print(cross_val_score(rfc,X_train,y_train,cv=5).mean())
# print(cross_val_score(xgbc,X_train,y_train,cv=5).mean())
# print(cross_val_score(svc,X_train,y_train,cv=5).mean())
#
#
# rfc.fit(X_train,y_train)
# rfc_y_predict=rfc.predict(X_test)
# rfc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':rfc_y_predict})
# rfc_submission.to_csv('F:/demo/kaggle/titanic/rfc_submission.csv',index=False)
#
#
# xgbc.fit(X_train,y_train)
# xgbc_y_predict=xgbc.predict(X_test)
# xgbc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':xgbc_y_predict})
# xgbc_submission.to_csv('F:/demo/kaggle/titanic/xgbc_submission.csv',index=False)
#
# svc.fit(X_train,y_train)
# svc_y_predict=svc.predict(X_test)
# svc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':svc_y_predict})
# svc_submission.to_csv('F:/demo/kaggle/titanic/svc_submission.csv',index=False)
#
#
# ##使用并行网格搜索的方式寻找更好的超参数组合
# params={'max_depth':range(2,15),'n_estimators':range(100,1100,200),'learning_rate':[0.05,0.1,0.2,0.25,0.3,0.5,0.7,0.75,0.8,1.0]}
#
#
# xgbc_best=XGBClassifier()
# gs=GridSearchCV(xgbc_best,params,n_jobs=-1,cv=5,verbose=1)
# gs.fit(X_train,y_train)
#
# print(gs.best_score_)
# print(gs.best_params_)
#
# xgbc_best_y_predict=gs.predict(X_test)
# xgbc_best_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':xgbc_best_y_predict})
# xgbc_best_submission.to_csv('F:/demo/kaggle/titanic/xgbc_best_submission.csv',index=False)
#
# rfc_best=RandomForestClassifier()
# gs=GridSearchCV(rfc_best,params,n_jobs=-1,cv=5,verbose=1)
# gs.fit(X_train,y_train)
#
# print(gs.best_score_)
# print(gs.best_params_)
#
# rfc_best_y_predict=gs.predict(X_test)
# rfc_best_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':rfc_best_y_predict})
# rfc_best_submission.to_csv('F:/demo/kaggle/titanic/rfc_best_submission.csv',index=False)