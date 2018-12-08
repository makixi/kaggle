# House Prices: Advanced Regression Techniques
##### 总结一下日常在做的特征工程：<br>
1. 通过现有数据填充缺失值<br>
2. 转换一些看起来很明确的数值变量<br>
3. 标签编码一些分类变量<br>

---

### 列属性分析
列属性  | 解释
---- | ---
SalePrice  | 房产的美元价格
MSSubClass  | 建筑的种类
MSZoning | 房屋所在分区的分类
LotFrontage  | 连接房屋的街道距离
LotArea | 平方英尺大小
Street  | 道路通行类型
Alley | 小路通行类型
LotShape | 房屋形状
LandContour | 房屋平整度
Utilities | 可用的公用事业类型
LotConfig | 批量配置
LandSlope | 房屋斜率
Neighborhood | Ames城市范围内的物理位置
Condition1 | 靠近主干道或铁路
Condition2 | 靠近主干道或铁路（如果存在第二条铁路）
BldgType | 住宅类型
HouseStyle | 住宅风格
OverallQual | 整体材料和表面质量
OverallCond | 总状态额定值
YearBuilt | 原始施工日期
YearRemodAdd | 改型日期
RoofStyle | 屋顶类型
RoofMatl | 屋顶材料
Exterior1st | 房屋外覆盖
Exterior2nd | 房屋外覆盖(如果不止一个)
MasVnrType | 表层砌体类型
MasVnrArea | 砌体面积
ExterQual | 外部材料质量
ExterCond | 外部材料的现状
Foundation | 基础类型
BsmtQual | 地下室高度
BsmtCond | 地下室现状
BsmtExposure | 花园式地下室
BsmtFinType1 | 地下室竣工面积质量
BsmtFinSF1 | 类型1完成的平方英尺
BsmtFinType2 | 第二成品面积的质量（如果存在）
BsmtFinSF2 | 类型2完成的平方英尺
BsmtUnfSF | 未完工的地下室平方英尺
TotalBsmtSF | 地下室总面积
Heating | 加热类型
HeatingQC | 加热质量与状态
CentralAir | 中央空调
Electrical | 电力系统
1stFlrSF | 一楼平方英尺
2ndFlrSF | 二楼平方英尺
LowQualFinSF | 低质量基金
GrLivArea | 高于（地面）生活面积平方英尺
BstmFullBath | 地下室全浴室
BstmHalfBath | 地下室半浴室
FullBath | 高档浴室
HalfBath | 高档半浴
Bedroom | 地下室层以上卧室数
Kitchen | 厨房数量
KitchenQual | 厨房质量
TotRmsAbvGrd | 总级别以上的房间（不包括浴室）
Functional | 家庭功能评级
Fireplaces | 壁炉数
FireplaceQu | 壁炉质量
GarageType | 车库位置
GarageYrBlt | 年车库建成
GarageFinish | 车库内饰
GarageCars | 车库容量
GarageArea | 车库英尺面积
GarageQual | 车库质量
GarageCond | 车库现状
PavedDrive | 铺砌车道
WoodDeckSF | 平方英尺木甲板面积
OpenPorchSF | 平方英尺开敞廊道面积
EnclosedPorch | 方块封闭廊道面积
3SsnPorch | 平方英尺三季门廊区
ScreenPorch | 平方英尺屏风面积
PoolArea | 游泳池面积
PoolQC | 游泳池质量
Fence | 围栏质量
MiscFeature | 其他类别未涵盖的杂项特征
MiscVal | 杂项价值
MoSold | 月售出
YrSold | 年售出
SaleType | 销售类型
SaleCondition | 销售现状



---

### 数据处理
首先通过作图探究一下数据中是否存在**离群点**。若有，就去掉，以免其对于整体的影响。<br>
*PS：离群点的移除不一定对于余下工程结果存在积极影响，这是一次尝试。并且这两个离群点离群很远*
```python
fig=plt.figure()
ax=fig.subplots()
ax.scatter(x=(train['1stFlrSF']+train['2ndFlrSF']+train['TotalBsmtSF']),y=train['SalePrice'])
plt.ylabel('SalePrice',fontsize=12)
plt.xlabel('TotalArea',fontsize=12)
plt.show()
#去掉离群点
train.drop(train[((train['1stFlrSF']+train['2ndFlrSF']+train['TotalBsmtSF'])>6000)&(train['SalePrice']<200000)].index,axis=0,inplace=True)
```

---

#### 重要数据关系正态分布

*线性模型会喜欢正态分布的数据，所以先对数据进行观测。再对其进行处理，使之更接近于正态分布*
```python
#观察与正态分布的距离
sns.distplot(train['SalePrice'],fit=norm)
(mu,sigma)=norm.fit(train['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.title("SalePrice distribution")
plt.ylabel("Frequency")
plt.show()


#qq plot
fig=plt.figure()
res=stats.probplot(train['SalePrice'],plot=plt)
plr.show()

train['SalePrice']=np.log2(train['SalePrice'])
```

---

#### 多缺失属性忽略

接下俩进行下一步数据分析。在表格中存在一些属性，几乎都是nan，这些属性对于接下来的分析归类没有作用，所以可以舍去。
```python
all_data_na=(all_data.isnull().sum())*100/all_data.shape[0]
all_data_na.drop(all_data_na[all_data_na==0].index,axis=0,inplace=True)
all_data_na.sort_values(ascending=False,inplace=True)
print(all_data_na)

all_data.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley'], axis=1, inplace=True)
```

---

#### 缺失数据填充

对于 None值的填充
```python
# 分离类别特征和数值特征
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
category_feats = all_data.dtypes[all_data.dtypes == 'object'].index
 
# 缺失值填充,类别特征缺失填充None，数值特征缺失填充均值
for col in category_feats:
    all_data[col] = all_data[col].fillna("None")
for col in numeric_feats:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
```

---

#### 字符类型与数值类型按需转换

```python

# 将数值特征转化为类别特征
for col in ('MSSubClass', 'OverallCond', 'YrSold', 'MoSold'):
    all_data[col] = all_data[col].astype(str)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC',  'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street',  'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# 类型=》数值
for col in cols:
    lbl=LabelEncoder()
    lbl.fit(list(all_data[col].values))
    all_data[col]=lbl.transform(list(all_data[col].values))
```

---

#### 数值特征正态化

[boxcox变换方法及实现应用](https://wenku.baidu.com/view/96140c8376a20029bd642de3.html)

```python
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness.head(10))


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
```

---

#### 标准化
[标准化处理](https://blog.csdn.net/bbbeoy/article/details/73662455)

---

### 建模

对于基于决策树的模型，调参的方法都是大同小异。一般都需要如下步骤：<br>
1.首先选择较高的学习率，大概0.1附近，这样是为了加快收敛的速度。这对于调参是很有必要的。<br>
2.对决策树基本参数调参<br>
3.正则化参数调参<br>
4.最后降低学习率，这里是为了最后提高准确率<br>

---

#### Lasso Regression
Lasso是在线性模型上加上了一个l1正则项<br>
[Lasso Regression](https://blog.csdn.net/daunxx/article/details/51596877)

---

#### Elastic Net
ElasticNet 是一种使用L1和L2先验作为正则化矩阵的线性回归模型.这种组合用于只有很少的权重非零的稀疏模型.<br>
可以使用 l1_ratio 参数来调节L1和L2的凸组合(一类特殊的线性组合)。 <br>
当多个特征和另一个特征相关的时候弹性网络非常有用。Lasso倾向于随机选择其中一个，而ElasticNet更倾向于选择两个. 

```python
#帮助elasticnet选参数
# 帮助elasticnet选择参数
alphas=np.logspace(-5, 1, 60)
train_errors = list()
for alpha in alphas:
    ent.set_params(alpha=alpha)
    ENet = make_pipeline(MinMaxScaler(), ent)
    train_errors.append(rmsle_cv(ENet).mean())

i_alpha_optim = np.argmin(train_errors)
alpha_optim = alphas[i_alpha_optim]
print("Optimal regularization parameter : %s" % alpha_optim)

plt.figure()
plt.semilogx(alphas, train_errors, label='Train')
plt.vlines(alpha_optim, plt.ylim()[0], np.max(train_errors), color='k',
           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left')
plt.ylim([0, 1.2])
plt.xlabel('Regularization parameter')
plt.ylabel('Performance')
plt.show()
```

---

#### KernelRidge
使用核技巧的岭回归（L2正则线性回归），它的学习形式和SVR（support vector regression）相同，但是两者的损失函数不同：KRR使用的L2正则均方误差；SVR使用的是带L2正则的ϵ-insensitive loss：max(0,|y−hθ(x)|−ϵ)<br>
KRR有近似形式的解，并且在中度规模的数据时及其有效率，由于KRR没有参数稀疏化的性能，因此速度上要慢于SVR（它的损失函数有利于得到稀疏化的解）。 <br>
KRR的最小二乘解：β=((K+λI)^-1)y，w=∑βiXi，这里的K是核函数。最小二乘解不适用于大规模数据。<br>
[Kernel Ridge Regression](https://blog.csdn.net/qsczse943062710/article/details/76021034)

---

#### GradientBoostingRegressor
[GradientBoostingRegressor](https://blog.csdn.net/u013395516/article/details/79809797?utm_source=blogxgwz7)<br>

1.learning_rate： <br>
控制每个树的作用有多大；值稍微小点好些；但如果太小则需要更多树； 默认0.1学习速率/步长0.0-1.0的超参数，每个树学习前一个树的残差的步长。<br>
2.n_estimators <br>
树的个数（弱学习器个数）；树太多容易过拟合，最好使用CV来调整以确定学习速率。默认100<br>
3.subsample <br>
训练某棵树时候，采用的样本的比重（往往使用80%）<br>
4.loss<br>
默认ls损失函数。'ls'是指最小二乘回归，lad'（最小绝对偏差），'huber'是两者的组合<br>
5.max_depth<br>
默认值为3每个回归树的深度，控制树的大小，也可用叶节点的数量max leaf nodes控制<br>
6.subsample<br>
默认为1,  用于拟合个别基础学习器的样本分数，选择子样本<1.0导致方差的减少和偏差的增加。<br>
7.min_samples_split<br>
默认为2, 生成子节点所需的最小样本数，如果是浮点数代表是百分比。<br>
8.min_samples_leaf<br>
默认为1, 叶节点所需的最小样本数，如果是浮点数代表是百分比<br>
9.max_features<br>
在寻找最佳分割点要考虑的特征数量auto全选/sqrt开方/log2对数/None全选/int自定义几个/float百分比<br>
10.max_leaf_nodes<br>
叶节点的数量 None不限数量<br>
11.min_impurity_split<br>
默认为1e-7, 停止分裂叶子节点的阈值<br>
12.verbose<br>
默认为0, 打印输出。大于1打印每棵树的进度和性能<br>
13.warm_start<br>
默认为False, True在前面基础上增量训练，False默认擦除重新训练 增加树<br>
14.random_state<br>
默认为0。随机种子-方便重现<br>

---

调参过程
```python
param_test2 = {'max_depth':range(4,8,1), 'min_samples_split':range(2,18,1)}
gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='lad', random_state =5, warm_start=True),
                                    param_grid = param_test2, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch2.fit(train.values, y_train)
print(gsearch2.param_grid,gsearch2.best_params_,gsearch2.best_score_)


param_test4 = {'max_features':range(1,20,1)}
gsearch4 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05,
                                   max_depth=4, 
                                   min_samples_leaf=2, min_samples_split=10,
                                   loss='lad', random_state =5, warm_start=True), 
                       param_grid = param_test4, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch4.fit(train.values, y_train)
print(gsearch4.param_grid, gsearch4.best_params_, gsearch4.best_score_)

param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05,
                                   max_depth=4, max_features=15,
                                   min_samples_leaf=2, min_samples_split=10,
                                   loss='lad', random_state =5, warm_start=True), 
                       param_grid = param_test5, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch5.fit(train.values, y_train)
print(gsearch5.param_grid, gsearch5.best_params_, gsearch5.best_score_)



model_xgb=make_pipeline)MinMaxScaler(),xgb.XGBRegressor(colsample_bytree=0.4603,gamma=0.0468,
                                            learning_rate=0.05,max_depth=4,
                                            min_child_weight=1.7817,n_estimators)

model_xgb =  make_pipeline(MinMaxScaler(),xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

model_lgb =  make_pipeline(RobustScaler(),lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

```

#### XGBRegressor

[XGBoost](https://blog.csdn.net/sb19931201/article/details/52557382)

1.'booster':'gbtree'<br>
2.'silent' <br>
设置成1则没有运行信息输出，最好是设置为0.<br>
3.'nthread'<br>
 cpu 线程数 默认最大<br>
4.'learning_rate'<br>
5.'min_child_weight'<br>
这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1分类而言，假设 h 在 0.01 附近，min_child_weight 为 1意味着叶子节点中最少需要包含 100个样本。<br>
这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。<br>
6.'max_depth' <br>
构建树的深度，越大越容易过拟合<br>
7.'gamma'<br>
树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。<br>
8.'subsample'<br>
随机采样训练样本<br>
9.'colsample_bytree'<br>
生成树时进行的列采样 <br>
10.'lambda':2<br>
控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。<br>
11.'alpha':0<br>
L1 正则项参数<br>
12.'scale_pos_weight'<br>
如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。<br>
12.'objective': 'multi:softmax'<br>
多分类的问题<br>
13.'num_class'<br>
类别数，多分类与 multisoftmax 并用<br>
14.'seed':1000<br>
随机种子<br>
15.'eval_metric': 'auc'<br>

---

#### LGBMRegressor

一些重要参数：<br>
1.max_depth:设置树的深度，深度越大越可能过拟合.<br>
2.num_leaves:用它调节树的复杂程度，大致换算关系num_leaves=2^(max_depth)，但是它的值应该小于2^(max_depth),否则可能会导致过拟合。<br>
3.min_data_in_leaf:它的值取决于样本训练的样例个数和num_leaves,将其设置的较大可以避免生成一个过深的树,但有可能导致欠拟合.<br>
4.min_sum_hessian_in_leaf:使一个节点分裂的最小海森值之和.<br>
5.feature_fraction:来进行特征的子抽样.这个参数可以用来防止过拟合及提高训练速度.
6.bagging_fraction+bagging_freq参数必须同时设置，bagging_fraction相当于subsample样本采样，可以使bagging更快的运行，同时也可以降拟合。bagging_freq默认0，表示bagging的频率，0意味着没有使用bagging，k意味着每k轮迭代进行一次bagging.<br>
7.正则化参数lambda_l1(reg_alpha), lambda_l2(reg_lambda)，是降低过拟合的<br>
8.learning_rate:较低的学习速率，以及使用更多的决策树n_estimators来训练数据<br>