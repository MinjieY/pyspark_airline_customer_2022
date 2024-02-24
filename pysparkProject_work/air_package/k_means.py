from matplotlib import pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id as mi
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
# 构建SparkSession执行环境入口对象
spark = SparkSession.builder. \
    appName('kmeans'). \
    master('local'). \
    getOrCreate()
sc = spark.sparkContext
data_LRFMC = spark.read.format('csv'). \
    option('sep', ','). \
        option('header', True). \
            option('encoding', 'gbk'). \
                load('D:\pycharm\pysparkProject_work\\air_package\data\\data_LRFMC_standard.csv')
data_LRFMC = data_LRFMC.withColumn("R",data_LRFMC['R'].astype("float"))\
    .withColumn("F",data_LRFMC['F'].astype("float")).\
    withColumn("M",data_LRFMC['M'].astype("float")).\
    withColumn("C",data_LRFMC['C'].astype("float")).\
    withColumn("L",data_LRFMC['L'].astype("float"))
data_LRFMC_pre = spark.read.format('csv'). \
    option('sep', ','). \
        option('header', True). \
            option('encoding', 'gbk'). \
                load('D:\pycharm\pysparkProject_work\\air_package\data\\data_LRFMC.csv')
data_LRFMC = data_LRFMC.withColumn("R",data_LRFMC['R'].astype("float"))\
    .withColumn("F",data_LRFMC['F'].astype("float")).\
    withColumn("M",data_LRFMC['M'].astype("float")).\
    withColumn("C",data_LRFMC['C'].astype("float")).\
    withColumn("L",data_LRFMC['L'].astype("float"))
#选取特征项，将特征项合并成向量
vecAss = VectorAssembler(inputCols = data_LRFMC.columns[:], outputCol = 'features')
data_ana = vecAss.transform(data_LRFMC).select('features')
print("==========data_ana特征项==========")
data_ana.show()
plt_pd = data_LRFMC.toPandas()

'''
用于画图的标准化数据
'''

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(plt_pd)
data = scaler.transform(plt_pd)
print("data[:5]=====标准化numpy")
print(data[:5])

airline_scale = data
# 可视化

#
# fig, ax = plt.subplots(1,1, figsize=(8,6))
# ax.plot(range(2,20), cost)
# ax.set_xlabel('k')
# ax.set_ylabel('cost')
# plt.show()

kMeans = KMeans(k=5,seed=1)
model = kMeans.fit(data_ana)
predic = model.transform(data_ana)
print("==========predic 特侦向量和分类==========")
print(type(predic))#特征向量预测表
predic.show(5)
result = model.transform(data_ana)
print(model.hasSummary)
print("==========聚类中心==========")
kmeans_center = model.clusterCenters()
print(kmeans_center)
print("==========kmeans_add聚类个数==========")
'''
[944, 890, 345, 701, 706]
'''
kmeans_add = model.summary.clusterSizes
print(kmeans_add)
#test data
from pyspark.ml.linalg import Vectors
testdata = spark.createDataFrame([
    (Vectors.dense([7.0,40.0,293678.0,1.2523144,2597.0]),)
],["test"])
print("==========第一行测试数据预测==========：")
testdata.show()
# model().transform(testdata).show()
testdata = testdata.withColumnRenamed("test","features")
model.transform(testdata).show()

'''
pyspark表连接
带有分类的新表：data_LRFMC_preL
+-----+-----+--------+---------+------+----------+
|    R|    F|       M|        C|     L|prediction|
+-----+-----+--------+---------+------+----------+
|  7.0|140.0|293678.0|1.2523144|2597.0|         2|
最后pandas统计画图
'''
#data_LRFMC_pre['type'] = predic['prediction']
id = mi()#用于列匹配
#data_change1.show()
predict_1 = predic
predict_1 = predict_1.withColumn('temp_col',id)   #add date_diff
data_LRFMC_preL = data_LRFMC_pre.withColumn('temp_col',id)#创建一个临时列用于合并
data_LRFMC_preL = data_LRFMC_preL.join(predict_1,data_LRFMC_preL.temp_col==predict_1.temp_col,'inner')\
    .drop(data_LRFMC_preL.temp_col).drop(predict_1.temp_col).drop(predict_1.features)
print("=============data_LRFMC_preL==============")
data_LRFMC_preL.show()
#这里又转pandas统计
import pandas as pd
import seaborn as sns
type_result = data_LRFMC_preL.toPandas()
type_des = type_result.groupby('prediction').describe()
print("==========typedes===========")
print(type_des)
type_des.to_csv("D:\pycharm\pysparkProject_work\\air_package\data\\type_des.csv",
                             index=False,encoding="GBK",header = True,sep=',')
data_LRFMC_preL.toPandas().to_csv("D:\pycharm\pysparkProject_work\\air_package\data\\type_result.csv",
                             index=False,encoding="GBK",header = True,sep=',')

'''#type结果可视化  小提琴图'''
#把原来的numpy转化成pandas dataframe
data_LRFMC_standardpd=pd.DataFrame(airline_scale,columns=data_LRFMC_preL.columns[:5])
print("===========data_LRFMC_standardpd=============")
print(data_LRFMC_standardpd)
#还是列拼接
predict_2 = predic
id2 = mi()
predict_2 = predict_2.withColumn('temp_col',id2)   #add date_diff
data_LRFMC_standardpd_pyspark = spark.createDataFrame(data_LRFMC_standardpd)#sparksql
data_LRFMC_standardpd_pysparkL = data_LRFMC_standardpd_pyspark.withColumn('temp_col',id2)#创建一个临时列用于合并
data_LRFMC_standardpd_pysparkL = data_LRFMC_standardpd_pysparkL.join(predict_2,data_LRFMC_standardpd_pysparkL.temp_col==predict_2.temp_col,'inner')\
    .drop(data_LRFMC_standardpd_pysparkL.temp_col).drop(predict_2.temp_col).drop(predict_2.features)
print("=============data_LRFMC_standardpd_pyspark特征标准化和类别拼接==============")

standard_pd_type= data_LRFMC_standardpd_pysparkL.toPandas()
pdcolumns = standard_pd_type.columns.tolist()
print("pdcolumns")
print(pdcolumns)
fig = plt.figure(figsize=(30, 18), dpi=256)  # 指定绘图对象宽度和高度
for i in range(5):
    plt.subplot(2, 3, i + 1)  # 2行3列子图
    ax = sns.violinplot(x="prediction", y=pdcolumns[i], width=0.8, saturation=0.9, lw=0.8, palette="Set2", orient="v",
                        inner="point", data=standard_pd_type)#dataframe没有get方法  ，跟pandas转来转去我要死了
    plt.xlabel((['customertype' + str(i) for i in range(5)]), fontsize=3)
    plt.ylabel(pdcolumns[i], fontsize=10)


sns.pairplot(data = standard_pd_type,height = 2.5,hue = 'prediction')

#客户特征折线图
cluster = kmeans_center
x=[1,2,3,4,5]
colors = ['red','pink','yellow','blue','green']
fig = plt.figure(figsize=(30, 25), dpi=256)  # 指定绘图对象宽度和高度
for i in range(5):
    plt.plot(x,cluster[i],label = 'cluster'+str(i),linewidth = 6-i,color = colors[i],
             marker = 'o')
plt.xlabel('featrues: 1.0->R    2.0->F    3.0->M    4.0->C    5.0->L')
plt.ylabel('values')
plt.show()



