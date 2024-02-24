from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import SQLContext
import pyspark
# 构建SparkSession执行环境入口对象
spark = SparkSession.builder. \
    appName('Second App'). \
    master('local'). \
    getOrCreate()
sc = spark.sparkContext
# 读取数据
dfread = spark.read.format('csv'). \
    option('sep', ','). \
        option('header', False). \
            option('encoding', 'GBK'). \
                load('D:\pycharm\pysparkProject_work\data\data.csv')
dfread.printSchema()
print("原数据表：")
dfread.show()
#1更换表头，
mapping = dict(zip(['_c1', '_c2','_c3', '_c4','_c5'], ['地区', '发病数','死亡数','发病率','死亡率']))
df1 = dfread.select([col(c).alias(mapping.get(c, c)) for c in dfread.columns])

print("原表列类型：")
df1.printSchema()

#2更改列的数据类型
df2 = df1.withColumn("死亡率",df1['死亡率'].astype("float"))\
    .withColumn("发病数",df1['发病数'].astype("int")).\
    withColumn("死亡数",df1['死亡数'].astype("int")).\
    withColumn("发病率",df1['发病率'].astype("float"))
#删除null行和列
#drop 删除空值列后非空值小于两个的行
dfwash = df2.drop('_c0').dropna(how='all',thresh=2)
#3将null值用0替换
df3 = dfwash.fillna(0)
print("修改后数据表列数据类型：")
df3.printSchema()
#4将新表保存为csv文件
print("清洗后的数据表：")
df3.show()
#转成pandas保存 隐藏序号列
dfresult = df3.toPandas()
print(dfresult)
dfresult.to_csv("D:\pycharm\pysparkProject_work\data\data_result1.csv", index=False,encoding="GBK",header = True,sep=',')
#直接保存
#df3.write.csv("D:\pycharm\pysparkProject_work\data\data_result2.csv", mode ='overwrite', encoding="GBK",header = True,sep=',')