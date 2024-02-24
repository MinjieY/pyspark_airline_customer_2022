try:
    from pyspark import SparkConf
    from pyspark import SparkContext

    print("Successfully imported Spark Modules")
except ImportError as e:
    print("Can not import Spark Modules", e)
print("Hello!world")