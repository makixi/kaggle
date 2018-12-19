import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.stats import norm,skew,boxcox_normmax
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

train = pd.read_csv('F:/demo/kaggle/houseprices/train.csv')
test = pd.read_csv('F:/demo/kaggle/houseprices/test.csv')


# print(train.columns.values)
# print(test.columns.values)

# print(train.shape)
# print(test.shape)

# print(train.info())
# print(test.info())

train_ID=train['Id']
test_ID=test['Id']

#train.drop("Id",axis=1,inplace=True)
#test.drop("Id",axis=1,inplace=True)

# fig=plt.figure()
# ax=fig.subplots()
# ax.scatter(x=(train['1stFlrSF']+train['2ndFlrSF']+train['TotalBsmtSF']),y=train['SalePrice'])
# plt.ylabel('SalePrice',fontsize=12)
# plt.xlabel('TotalArea',fontsize=12)
# plt.show()

#去掉离群点
train.drop(train[((train['1stFlrSF']+train['2ndFlrSF']+train['TotalBsmtSF'])>6000)&(train['SalePrice']<200000)].index,axis=0,inplace=True)

# fig=plt.figure()
# ax=fig.subplots()
# ax.scatter(x=(train['1stFlrSF']+train['2ndFlrSF']+train['TotalBsmtSF']),y=train['SalePrice'])
# plt.ylabel('SalePrice',fontsize=12)
# plt.xlabel('TotalArea',fontsize=12)
# plt.show()

train['SalePrice']=np.log1p(train['SalePrice'])

# #观察与正态分布的距离
# sns.distplot(train['SalePrice'],fit=norm)
# (mu,sigma)=norm.fit(train['SalePrice'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
# plt.title("SalePrice distribution")
# plt.ylabel("Frequency")
# plt.show()


# #qq plot
# fig=plt.figure()
# res=stats.probplot(train['SalePrice'],plot=plt)
# plt.show()



y_train=train.SalePrice.values
train.drop(['SalePrice'],axis=1,inplace=True)
all_data=pd.concat((train,test)).reset_index(drop=True)
# print(all_data.columns.values)
# print(all_data.shape)

ntrain=train.shape[0]

# all_data_na=(all_data.isnull().sum())*100/all_data.shape[0]
# all_data_na.drop(all_data_na[all_data_na==0].index,axis=0,inplace=True)
# all_data_na.sort_values(ascending=False,inplace=True)
# print(all_data_na)

#90%以上缺失的列值去掉
all_data.drop([ 'MiscFeature', 'Alley'], axis=1, inplace=True)

# print(all_data.info())

# 分离类别特征和数值特征
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
category_feats = all_data.dtypes[all_data.dtypes == 'object'].index
 
# 缺失值填充,类别特征缺失填充None，数值特征缺失填充均值
for col in category_feats:
    all_data[col] = all_data[col].fillna("None")
for col in numeric_feats:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
 
 #print(all_data.info())

# 将数值特征转化为类别特征
for col in ('MSSubClass', 'OverallCond', 'YrSold', 'MoSold'):
    all_data[col] = all_data[col].astype(str)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'PoolQC', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# 类型=》数值
for col in cols:
    lbl=LabelEncoder()
    lbl.fit(list(all_data[col].values))
    all_data[col]=lbl.transform(list(all_data[col].values))


# print(all_data.head(30))

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# 数值特征正态化
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
# print(skewness.head(10))

skewness = skewness[abs(skewness) > 0.75]
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

#all_data[skewed_features] = np.log1p(all_data[skewed_features])


all_data = pd.get_dummies(all_data)
train = all_data[:ntrain]
test = all_data[ntrain:]

#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


la=Lasso(alpha =0.000335292414924956, random_state=1)
lasso = make_pipeline(MinMaxScaler(), la)
score = rmsle_cv(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

ENet = make_pipeline(MinMaxScaler(), ElasticNet(alpha=0.0004237587160604063, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

krr=KernelRidge(alpha=0.37693909753883637, kernel='polynomial', degree=2.061224489795918, coef0=3)
KRR = make_pipeline(RobustScaler(),krr)
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



GBoost =  make_pipeline(RobustScaler(),GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05,
                                   max_depth=4, max_features=15,subsample=0.8,
                                   min_samples_leaf=2, min_samples_split=10,
                                   loss='lad', random_state =5, warm_start=True))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# param_test2 = {'max_depth':range(4,8,1), 'min_samples_split':range(2,18,1)}
# gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05,
#                                    max_depth=4, max_features='sqrt',
#                                    min_samples_leaf=15, min_samples_split=10,
#                                    loss='lad', random_state =5, warm_start=True),
#                                     param_grid = param_test2, scoring='neg_mean_squared_error',iid=False, cv=5)
# gsearch2.fit(train.values, y_train)
# print(gsearch2.param_grid,gsearch2.best_params_,gsearch2.best_score_)


# param_test4 = {'max_features':range(1,20,1)}
# gsearch4 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05,
#                                    max_depth=4, 
#                                    min_samples_leaf=2, min_samples_split=10,
#                                    loss='lad', random_state =5, warm_start=True), 
#                        param_grid = param_test4, scoring='neg_mean_squared_error',iid=False, cv=5)
# gsearch4.fit(train.values, y_train)
# print(gsearch4.param_grid, gsearch4.best_params_, gsearch4.best_score_)

# param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
# gsearch5 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05,
#                                    max_depth=4, max_features=15,
#                                    min_samples_leaf=2, min_samples_split=10,
#                                    loss='lad', random_state =5, warm_start=True), 
#                        param_grid = param_test5, scoring='neg_mean_squared_error',iid=False, cv=5)
# gsearch5.fit(train.values, y_train)
# print(gsearch5.param_grid, gsearch5.best_params_, gsearch5.best_score_)


model_xgb =  make_pipeline(MinMaxScaler(),xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=3, n_estimators=2170,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# param_test1={'gamma': [0.0468,0.0469,0.047,0.0467,0.0466]}
# gsearch1=GridSearchCV(estimator=xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
#                              learning_rate=0.05, max_depth=3,
#                              min_child_weight=3, n_estimators=2170,
#                              reg_alpha=0.4640, reg_lambda=0.8571,
#                              subsample=0.5213, silent=1,
#                              random_state =7, nthread = -1),
#                     param_grid=param_test1,scoring='neg_mean_squared_error',iid=False,cv=5)
# gsearch1.fit(train.values,y_train)
# print(gsearch1.param_grid, gsearch1.best_params_, gsearch1.best_score_)




model_lgb =  make_pipeline(RobustScaler(),lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=1440,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq =5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,max_depth=2,reg_alpha=0.3,reg_lambda=0.08,
                              min_data_in_leaf =7, min_sum_hessian_in_leaf = 0.001))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))



# param_test1={'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],    'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]}
# gsearch1=GridSearchCV(estimator=lgb.LGBMRegressor(objective='regression',num_leaves=5,
#                               learning_rate=0.1, n_estimators=720,
#                               max_bin = 55, bagging_fraction = 0.8,
#                               bagging_freq = 5, feature_fraction = 0.2319,
#                               feature_fraction_seed=9, bagging_seed=9,max_depth=2,
#                               min_data_in_leaf =6, min_sum_hessian_in_leaf = 11),
#                     param_grid=param_test1,scoring='neg_mean_squared_error',iid=False,cv=5)
# gsearch1.fit(train.values,y_train)
# print(gsearch1.param_grid, gsearch1.best_params_, gsearch1.best_score_)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),meta_model = lasso)
#stacked_averaged_models = StackingAveragedModels(base_models = (KRR, GBoost, lasso),meta_model = ENet)
# stacked_averaged_models = StackingAveragedModels(base_models = (KRR, ENet, lasso),meta_model = GBoost)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))


'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +xgb_train_pred*0.15 + lgb_train_pred*0.15 ))

ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('F:/demo/kaggle/houseprices/submission.csv',index=False)