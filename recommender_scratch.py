import pyspark
import pyspark.sql
from pyspark.sql import SparkSession
from pyspark import SparkFiles

# Data from https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions?select=interactions_train.csv

spark = SparkSession.builder.master("local[1]")\
    .appName("recommender_project")\
    .getOrCreate()

url = "C:/Users/stanl/Downloads/interactions_train.csv"
df = spark.read.format("csv").load(url)

print(df.head(5))