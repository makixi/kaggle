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
