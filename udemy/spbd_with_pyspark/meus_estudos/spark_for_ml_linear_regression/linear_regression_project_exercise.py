from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('cruise').getOrCreate()
df = spark.read.csv('cruise_ship_info.csv',inferSchema=True,header=True)
df.printSchema()
df.show()
df.describe().show()
df.groupBy('Cruise_line').count().show()

from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Cruise_line", outputCol="cruise_cat")
indexed = indexer.fit(df).transform(df)
indexed.head(5)

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

print(indexed.columns)

assembler = VectorAssembler(
  inputCols=['Age',
             'Tonnage',
             'passengers',
             'length',
             'cabins',
             'passenger_density',
             'cruise_cat'],
    outputCol="features")

output = assembler.transform(indexed)

output.select("features", "crew").show()
final_data = output.select("features", "crew")
final_data.show()

train_data,test_data = final_data.randomSplit([0.7,0.3])

from pyspark.ml.regression import LinearRegression
# Create a Linear Regression Model object
lr = LinearRegression(labelCol='crew')

# Fit the model to the data and call this model lrModel
lrModel = lr.fit(train_data)

# Print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))

test_results = lrModel.evaluate(test_data)

print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
print("R2: {}".format(test_results.r2))

# R2 of 0.86 is pretty good, let's check the data a little closer
from pyspark.sql.functions import corr

df.select(corr('crew','passengers')).show()

df.select(corr('crew','cabins')).show()