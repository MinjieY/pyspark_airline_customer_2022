
from pyspark.sql import SparkSession
from matplotlib import pyplot as plt
import seaborn as sns
import pyspark.sql.functions as F
from pyspark.sql.functions import datediff
from pyspark.sql.functions import monotonically_increasing_id as mi
import findspark

findspark.init()
'''
针对航空客户数据从数据清洗、属性归纳与数据变换入手进行数据预处理
'''
# 构建SparkSession执行环境入口对象
spark = SparkSession.builder. \
    appName('air App2'). \
    master('local'). \
    getOrCreate()
sc = spark.sparkContext
# 读取数据
dfread = spark.read.format('csv'). \
    option('sep', ','). \
        option('header', True). \
            option('encoding', 'GBK'). \
                load('D:\pycharm\pysparkProject_work\\air_package\data\\air_data.csv')
'''
1.1、数据异常值处理
通过数据的探索分析发现数据中存在票价为空值、票价为0、
折扣率最小值为0、飞行公里数大于0的记录。由于这块的数
据所占比重较小，故采用丢弃的处理办法
'''
# 删除年龄中的异常值
data_remove_agemg = dfread.where('AGE<100')
# 去除票价为空的记录
colsnull = ['SUM_YR_1','SUM_YR_2']
data_not_null = data_remove_agemg.dropna(how='any',subset=colsnull)
#data_not_null.show(135)
# 平均折扣率不为0且总飞行公里数大于0的记录
data_not_null.registerTempTable("data_not_null")#生成临时表
sql_remove = "select * from data_not_null WHERE (avg_discount!=0 AND avg_discount>0)"
data_remove_diff = spark.sql(sql_remove)
#data_remove_diff.show()

'''
1.2、缺失值处理
发现有4个类别型数据：WORK_CITY，WORK_PROVINCE ，WORK_COUNTRY ，GENDER中缺失值
1个连续型数据：AGE有缺失值
由相关性图可以看出，年龄与其他属性的相关性低，且年龄分布较为集中，
因此这里采用均值填充的方式
'''
#查找age列的平均值，再填充进去
age_mean_frame = data_remove_diff.select(F.avg(data_remove_diff['AGE']))
age_mean = age_mean_frame.columns[0]
data_fillna = data_remove_diff.fillna(age_mean,'AGE')
print("data_fillna")
data_fillna.show(8)



'''
2、选择有用的列数据进行分析
'''

airline_selection = data_fillna[['FFP_DATE','LOAD_TIME','LAST_TO_END',
                                 'FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]
# airline_selection.show()
airline_selection.describe().show()
'''
#构造入会时长指标，并对数据进行标准化
# 构造L 单位为天数
'''
L = airline_selection.select(datediff(airline_selection['LOAD_TIME'],
                                      airline_selection['FFP_DATE']).alias('date_diff'))

# 构造LRFMC指标
#print(type(L))
id = mi()#用于列匹配
data_change1 = airline_selection.drop('LOAD_TIME','FFP_DATE')#delete load_diff ffp_date
#data_change1.show()
LL = L.withColumn('temp_diff',id)   #add date_diff
data_change2 = data_change1.withColumn('temp_diff',id)#创建一个临时列用于合并
data_change2 = data_change2.join(LL,LL.temp_diff==data_change2.temp_diff,'inner')\
    .drop(LL.temp_diff).drop(data_change2.temp_diff)
#直接创建新列并设置数据类型 便于后续分析
data_change2 = data_change2.withColumn("R",data_change2['LAST_TO_END'].astype("float"))\
    .withColumn("F",data_change2['FLIGHT_COUNT'].astype("float")).\
    withColumn("M",data_change2['SEG_KM_SUM'].astype("float")).\
    withColumn("C",data_change2['avg_discount'].astype("float"))\
    .withColumn("L",data_change2['date_diff'].astype("float"))
data_LRFMC = data_change2.drop('LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount','date_diff')
print("data_lrfmc_des")
data_LRFMC.describe().show()
data_LRFMC.toPandas().to_csv("D:\pycharm\pysparkProject_work\\air_package\data\data_LRFMC.csv",
                             index=False,encoding="GBK",header = True,sep=',')
print("data_colums")
print(data_LRFMC.columns)
#画箱线图
pd_tmp = data_LRFMC.toPandas()
print("data_pd_tmp")
print(pd_tmp)
column = pd_tmp.columns.tolist() # 列表头
fig = plt.figure(figsize=(22, 12), dpi=75)  # 指定绘图对象宽度和高度
for i in range(5):
    plt.subplot(2,3, i + 1)  # 2行3列子图
    sns.boxplot(data=pd_tmp[column[i]], orient="v",width=0.5)  # 箱式图
    plt.ylabel(column[i], fontsize=36)
    #核密度图
column = pd_tmp.columns.tolist() # 列表头
fig = plt.figure(figsize=(22, 12), dpi=75)  # 指定绘图对象宽度和高度
for i in range(5):
    plt.subplot(2,3, i + 1)  # 2行3列子图
    ax = sns.kdeplot(data=pd_tmp[column[i]],shade= True)
    plt.ylabel(column[i], fontsize=36)
    #热力图
corr = plt.subplots(figsize = (8,6))
corr= sns.heatmap(pd_tmp[column].corr(),annot=True,square=True,cmap='Blues')
# plt.show()
#pandas方便标准化  建立建模数据集
#df_model为标准化后的
'''标准化
'''
df_model=(pd_tmp-pd_tmp.mean(axis=0))/pd_tmp.std(axis=0)
print("df_model and type")
print(df_model,type(df_model))
df_model.to_csv("D:\pycharm\pysparkProject_work\\air_package\data\data_LRFMC_standard.csv",
                             index=False,encoding="GBK",header = True,sep=',')
#初步看一下聚为几类？


# k取 2-19 时 获取对应 cost 来确定 k 的最优取值
# 构建模型，确定k的值
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#判断据类数
scaler = StandardScaler()
data_np = data_LRFMC.toPandas().values
scaler.fit(data_np)
data = scaler.transform(data_np)
print("data")
print(data[:5])

sse = []
airline_scale = data
for k in range(1,10):
    model = KMeans(n_clusters=k, random_state=123, n_init=20)
    model.fit(airline_scale)
    sse.append(model.inertia_)
print("sse")
print(sse)
fig = plt.figure(figsize=(30, 25), dpi=256)#聚类数目选择
plt.plot(range(1, 10), sse, 'o-')
plt.axhline(sse[4], color='k', linestyle='--', linewidth=1)
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('K-means Clustering')
plt.show()


