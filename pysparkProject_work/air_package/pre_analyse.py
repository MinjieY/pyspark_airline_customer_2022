import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pyspark.sql import SparkSession
import missingno as mg
import findspark
findspark.init()
# 构建SparkSession执行环境入口对象
spark = SparkSession.builder. \
    appName('air App'). \
    master('local'). \
    getOrCreate()
sc = spark.sparkContext
# 读取数据
dfread = spark.read.format('csv'). \
    option('sep', ','). \
        option('header', True). \
            option('encoding', 'GBK'). \
                load('D:\pycharm\pysparkProject_work\\air_package\data\\air_data.csv')
print("原数据表描述：")

dfread.printSchema()
#统计缺失数据 倒序排列
dfread.toPandas().isnull().sum().sort_values(ascending=False)
dfread.describe().show()
des = dfread.describe()
print(type(des))
des.toPandas().to_csv("D:\pycharm\pysparkProject_work\\air_package\data\des_air_data.csv", index=False,encoding="GBK",header = True,sep=',')

for col in dfread.columns:
    print(col, dfread.filter(dfread[col].isNull()).count()/dfread.count())

# 缺失值可视化

dfpd = dfread.toPandas()#转成pandas方便画图
# print(dfpd)

mg.bar(dfpd,labels=1)
plt.show()

# 数据类别统计分析
print(dfpd.dtypes.value_counts())

# 连续型变量空值可视化
num_columns = dfpd.loc[:,dfpd.dtypes != object].columns
for var in num_columns:
    fig,ax = plt.subplots(figsize=(5,5))
    sns.boxplot(dfpd[var],orient='v')
    ax.set_xlabel(var)
'''
# 绘制各年份入会人数趋势图
# 将时间字符串转换为日期
'''
ffp = dfpd['FFP_DATE'].apply(lambda x : datetime.strptime(x,'%Y-%m-%d'))
# 提取入会年份

ffp_year = ffp.map(lambda x : x.year)
# 统计人数

ffp_year_count = ffp_year.value_counts()
plt.rcParams['font.sans-serif'] = [u'simHei']   # 显示中文
plt.rcParams['axes.unicode_minus'] = False      # 解决负号问题

plt.figure(figsize=(10,6))
plt.hist(ffp_year,bins='auto')
plt.xlabel('时间年份')
plt.ylabel('入会人数')
plt.title('入会人数变化趋势图')

#入会性别比例
gender =  dfpd['GENDER']
# 统计男女比例
gender_count = gender.value_counts()
gender_count = gender_count/gender_count.sum()
plt.figure(figsize=(5,5))
color = ['blue', 'red']
plt.pie(gender_count,labels=['男','女'],colors=color,autopct='%1.1f%%')
plt.title('入会性别比例',fontsize=20)

#会员卡级别—会员卡级别统计图
ffp_tier = dfpd['FFP_TIER']
# 会员卡级别统计
ffp_tier_count = ffp_tier.value_counts()
plt.figure(figsize=(6,6))
plt.hist(ffp_tier,bins='auto',alpha=0.8)
plt.xlabel('会员卡级别',fontsize=8)
plt.ylabel('人数',fontsize=8)
plt.title('会员卡级别分布',fontsize=10)

#年龄
age = dfpd['AGE'].dropna()
# 绘制箱型图
plt.figure(figsize=(5,5))
sns.boxplot(age,orient='h',flierprops = {'marker':'o',#异常值形状
                          'markerfacecolor':'red',#形状填充色
                          'color':'black',#形状外廓颜色
                         })
plt.xlabel('会员年龄',fontsize=10)
plt.title('会员年龄分布',fontsize=10)


#消费信息分析

#1飞行次数与观测窗口内的总飞行公里数
flight_count = dfpd['FLIGHT_COUNT']
# 统计飞行次数
print(flight_count.value_counts())
seg_km_sum = dfpd['SEG_KM_SUM']
# 统计飞行里程数
print(seg_km_sum.value_counts())
fig,ax = plt.subplots(1,2,figsize=(16,8))
sns.boxplot(flight_count,orient='h',ax=ax[0])
ax[0].set_xlabel('飞行次数',fontsize=15)
ax[0].set_title('会员飞行次数分布箱型图',fontsize=20)

sns.boxplot(seg_km_sum,orient='h',ax=ax[1])
ax[1].set_xlabel('飞行里程数',fontsize=15)
ax[1].set_title('会员飞行里程数分布箱型图',fontsize=20)
#2


#2票价收入
sum_yr = dfpd['SUM_YR_1'] + dfpd['SUM_YR_2']
print("sum_yr")
print(sum_yr.value_counts())
plt.figure(figsize=(10,10))
sns.boxplot(sum_yr,orient='h')
plt.xlabel('票价收入',fontsize=15)
plt.title('会员票价收入分布',fontsize=20)


#3平均时间间隔统计
avg_interval = dfpd['AVG_INTERVAL']
print(avg_interval.value_counts())
plt.figure(figsize=(6,6))
sns.boxplot(avg_interval,orient='h')
plt.xlabel('平均乘机时间间隔',fontsize=8)
plt.title('会员平均乘机时间间隔分布箱型图',fontsize=10)
# 4 最后一次乘机时间至观测窗口时长
last_to_end = dfpd['LAST_TO_END']
plt.figure(figsize=(6,6))
sns.boxplot(last_to_end,orient='h')
plt.xlabel('最后一次乘机时间至观测窗口时长',fontsize=9)
plt.title('客户最后一次乘机时间至观测窗口时长箱型图',fontsize=10)

#------积分信息分布
exchange_count = dfpd['EXCHANGE_COUNT']
# 1统计积分兑换次数
print(exchange_count.value_counts())
plt.figure(figsize=(5,5))
plt.hist(exchange_count,edgecolor='r',bins='auto',align='mid')
plt.xlabel('积分兑换次数',fontsize=8)
plt.ylabel('会员人数',fontsize=8)
plt.title('会员卡积分兑换次数分布直方图',fontsize=10)


#2总累计积分
point_sum= dfpd['Points_Sum']
# 统计总累计积分
print(point_sum.value_counts())
plt.figure(figsize=(5,5))
sns.boxplot(point_sum,orient='h')
plt.xlabel('总累计积分',fontsize=15)
plt.title('会员总累计积分分布箱型图',fontsize=20)
plt.show()
