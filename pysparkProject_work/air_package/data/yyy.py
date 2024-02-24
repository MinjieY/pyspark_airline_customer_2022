from pyspark.ml.feature import VectorAssembler
from matplotlib import pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id as mi
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import findspark
findspark.init()
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
                load('D:\pycharm\pysparkProject_work\\air_package\data\\data_LRFMC.csv')
data_LRFMC = data_LRFMC.withColumn("R",data_LRFMC['R'].astype("float"))\
    .withColumn("F",data_LRFMC['F'].astype("float")).\
    withColumn("M",data_LRFMC['M'].astype("float")).\
    withColumn("C",data_LRFMC['C'].astype("float")).\
    withColumn("L",data_LRFMC['L'].astype("float"))
data_LRFMC.show(5)


vecAss = VectorAssembler(inputCols = data_LRFMC.columns[1:], outputCol = 'features')
df_featrues = vecAss.transform(data_LRFMC).select('index', 'features')
df_featrues.show(3)

from pyspark.ml.feature import VectorAssembler

assemble = VectorAssembler(inputCols=['towerKills', 'inhibitorKills',
                                      'baronKills', 'dragonKills', 'riftHeraldKills'
                                      ], outputCol='features')

assembled_data = assemble.transform(dfseries)
assembled_data.show(2)
# 向量化


from pyspark.ml.feature import StandardScaler
 #标准化
scale=StandardScaler(inputCol='features',outputCol='standardized')
data_scale=scale.fit(assembled_data)
data_scale_output=data_scale.transform(assembled_data)
data_scale_output.show(3)

# test
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

silhouette_score = []
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized', \
                                metricName='silhouette', distanceMeasure='squaredEuclidean')

for i in range(2, 10):
    KMeans_algo = KMeans(featuresCol='standardized', k=i)
    KMeans_fit = KMeans_algo.fit(data_scale_output)
    output = KMeans_fit.transform(data_scale_output)
    score = evaluator.evaluate(output)
    silhouette_score.append(score)
    print(" Score:", score)

# 可视化轮廓分数
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,10),silhouette_score)
ax.set_xlabel('k')
ax.set_ylabel('cost')