# pyspark_airline_customer_2022
big data
## 编程环境:pycharm+miniconda+pyspark(spark-3.3.1,python3.9);<br>
• 简介: 使用pyspark对获取到的航空公司原始数据进行预处理，读取csv文件以后主要用
到numpy和dataframe对数据表进行类型变换，loc、query等方法进行行和列的条件筛
选，选用合适的方法对原始数据进行初步处理（去除空值，重复数据，均值填充，异常
值去除等等），利用内置的sql库辅助进行数据库的查询，根据LRFMC模型提取航空公司
的部分数据，基于大数据集的特点，利用 K-means 聚类方法，以均方差确定指标，划分航空公司不同类别的客户
群体，最后调用plot、seaborn等进行可视化，提出针对性的营销策略。
