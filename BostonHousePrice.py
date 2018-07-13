import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("boston_housing.csv")
'''
本例子使用波士顿房价模型就行练习
'''

# print(data.head())
# print(data.info())
# print(data.isnull().sum())#查看是否有空值

'''
探索数据

查看数据各特征的分布，以及特征之间是否存在相关关系等冗余。

我们可以借用可视化工具来直观感觉数据的分布。

在Python中，有很多数据可视化途径。 Matplotlib非常强大，也很复杂，不易于学习。
 Seaborn是在matplotlib的基础上进行了更高级的API封装，从而使得作图更加容易，
 在大多数情况下使用seaborn就能做出很具有吸引力的图，而使用matplotlib就能制作具有更多特色的图。

'''
# print(data.describe())#此处得到各属性的样本数目、均值、标准差、最小值、1/4分位数（25%）、中位数（50%）、3/4分位数（75%）、最大值 可初步了解各特征的分布


#单变量分布分析

# 目标y（房屋价格）的直方图／分布
# fig=plt.figure()
# sns.distplot(data.MEDV.values,bins=30,kde=True)
# plt.xlabel('Median value of owner-occupied homes',fontsize=12)
# plt.show()

# 单个特征散点图
# plt.scatter(range(data.shape[0]),data['MEDV'].values,color='purple')#data.shape[0]=506
# plt.title('Distribution of Price')
# plt.xlabel('the number of the sample')
# plt.show()
#可以看出，数据大多集中在均值（22.5）附近，和正态分布比较接近。
# 但最大值50的样本数目偏多，可能是原始数据将所有大于50的样本的值都设置为50（猜测），
# 在模型训练时也可以考虑将y等于50的样本当成outliers（离群点）去掉。

#删除大于50的样本
# data2=data[data.MEDV < 50]
# print(data2.shape)

# 输入属性的直方图／分布
# fig=plt.figure()
# sns.distplot(data.ZN.values,bins=30,kde=False)
# plt.ylabel('samples count')
# plt.xlabel('proportion of residential land zoned',fontsize=12)
# plt.show()


# fig=plt.figure()
# plt.hist(data.INDUS.values,bins=30)
# plt.xlabel('proportion of non-retail business areas',fontsize=13)
# plt.show()
# #犯罪率特征的分布是长尾分布，和指数分布比较接近。大部分城镇的犯罪率很低，
# # 极少数样本的犯罪率高。从常理看，该数值应该比较准确，可以不予处理。

# sns.countplot(data.CHAS, order=[0, 1]);
# plt.xlabel('Charles River');
# plt.ylabel('Number of occurrences');
# plt.show()


# fig = plt.figure()
# sns.distplot(data.NOX.values, bins=30, kde=False)
# plt.xlabel('nitric oxides concentratio', fontsize=12)
# plt.show()


# fig = plt.figure()
# sns.distplot(data.RM.values, bins=30, kde=False)
# plt.xlabel('average number of rooms per dwelling', fontsize=12)
# plt.show()


# fig = plt.figure()
# sns.distplot(data.AGE.values, bins=30, kde=False)
# plt.xlabel('proportion of owner-occupied units built prior to 1940', fontsize=12)
# plt.show()


# fig = plt.figure()
# sns.distplot(data.DIS.values, bins=30, kde=False)
# plt.xlabel('weighted distances to five Boston employment centres', fontsize=12)
# plt.show()


# fig = plt.figure()
# sns.distplot(data.RAD.values, bins=20, kde=False)
# plt.xlabel('index of accessibility to radial highways', fontsize=12)
# plt.show()


# sns.countplot(data.RAD);
# plt.xlabel('index of accessibility to radial highways');
# plt.show()


# fig = plt.figure()
# sns.distplot(data.TAX.values, bins=20, kde=False)
# plt.xlabel('full-value property-tax rate per $10,000', fontsize=12)
# plt.show()


# sns.countplot(data.PTRATIO);
# plt.xlabel('pupil-teacher ratio by town');
# plt.show()



#
# fig = plt.figure()
# sns.distplot(data.B.values, bins=30, kde=False)
# plt.xlabel('proportion of blacks', fontsize=12)
# plt.show()


# fig = plt.figure()
# sns.distplot(data.LSTAT.values, bins=30, kde=False)
# plt.xlabel('lower status of the population', fontsize=12)
# plt.show()




# 两两特征之间的相关性
#计算相关系数，通常认为相关系数大于0.5的为强相关
cols=data.columns
data_corr=data.corr().abs()
# print(data_corr)
# print(data_corr.shape)
plt.subplots(figsize=(13,9))
sns.heatmap(data_corr,annot=True)#这个好玩，有意思
#mask unimportant features
# sns.heatmap(data_corr,mask=data_corr < 1,cbar=False)
plt.savefig('house_coor.png')
plt.show()

#Set the threshold to select only highly correlated attributes
threshold = 0.5
# List of pairs along with correlation above threshold
corr_list = []
#size = data.shape[1]# 14
size = data_corr.shape[0]#506

#Search for the highly correlated pairs
for i in range(0, size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print("%s and %s = %.2f" % (cols[i],cols[j],v))

# Scatter plot of only the highly correlated pairs
for v,i,j in s_corr_list:
    sns.pairplot(data, size=6, x_vars=cols[i],y_vars=cols[j] )
    plt.show()































