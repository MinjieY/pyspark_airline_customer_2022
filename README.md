# pyspark_airline_customer_2022

<br>

### 概述
• 编程环境:pycharm+miniconda+pyspark(spark-3.3.1,python3.9);<br>
• 简介: 使用pyspark对获取到的航空公司原始数据进行预处理，读取csv文件以后主要用
到numpy和dataframe对数据表进行类型变换，loc、query等方法进行行和列的条件筛
选，选用合适的方法对原始数据进行初步处理（去除空值，重复数据，均值填充，异常
值去除等等），利用内置的sql库辅助进行数据库的查询，根据LRFMC模型提取航空公司
的部分数据，基于大数据集的特点，利用 K-means 聚类方法，以均方差确定指标，划分航空公司不同类别的客户
群体，最后调用plot、seaborn等进行可视化，提出针对性的营销策略。
<br>
### /air_package
### Python源文件<br>
[1]. Pre_analyse.py:源数据文件探索性分析、源数据分析可视化；<br>
[2]. K_means.py:kmeans聚类求解，聚类结果可视化；<br>
[3]. Pre_resolve:数据清洗，数据预处理，聚类数探索。<br>
  ####  /data 文件
[1] type_result.csv：客户聚类结果及对应特征文件；<br>
[2] data_LRFMC_standard.csv：客户特征标准化dataframe；<br>
[3] data_LRFMC.csv：客户特征dataframe；<br>
[4] type_des.csv：分类结果数据描述；<br>
[5] des_air_data.csv :源数据描述 <br>
 #### /_fina_paper
 pyspark_paper:课程论文/报告
