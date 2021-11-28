import pyspark
import pyspark.sql
from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


# Data from https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions?select=interactions_train.csv

spark = SparkSession.builder.master("local[1]")\
    .appName("recommender_project")\
    .getOrCreate()

url = "C:/Users/stanl/Downloads/interactions_train.csv"
df = spark.read\
    .option("header", True)\
    .option("inferSchema", True)\
    .format("csv").load(url)

train_data = (df
    .select(
        'u', # unique user id
        'i', # unique recipe id
        'rating', # rating
    )
).cache()

(training, test) = train_data.randomSplit([0.8, 0.2])

als = ALS(maxIter=2, regParam=0.01,
          userCol="u", itemCol="i", ratingCol="rating",
          coldStartStrategy="drop",
          implicitPrefs=True)
model = als.fit(training)

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(5)
print("Recommendation count = " + str(userRecs.count()))