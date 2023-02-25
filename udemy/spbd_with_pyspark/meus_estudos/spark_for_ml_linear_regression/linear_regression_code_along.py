from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('lr_example').getOrCreate()

from pyspark.ml.regression import LinearRegression

# Use Spark to read in the Ecommerce Customers csv file.
data = spark.read.csv("Ecommerce_Customers.csv",inferSchema=True,header=True)

data.printSchema()

data.show()

data.head()

for item in data.head():
    print(item)

# A few things we need to do before Spark can accept the data!
# It needs to be in the form of two columns
# ("label","features")

# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

data.columns

assembler = VectorAssembler(
    inputCols=["Avg Session Length", "Time on App",
               "Time on Website",'Length of Membership'],
    outputCol="features")

output = assembler.transform(data)

output.select("features").show()

output.show()

final_data = output.select("features",'Yearly Amount Spent')

train_data,test_data = final_data.randomSplit([0.7,0.3])

train_data.describe().show()

test_data.describe().show()

# Create a Linear Regression Model object
lr = LinearRegression(labelCol='Yearly Amount Spent')

# Fit the model to the data and call this model lrModel
lrModel = lr.fit(train_data,)

# Print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))

test_results = lrModel.evaluate(test_data)

# Interesting results....
test_results.residuals.show()

unlabeled_data = test_data.select('features')

predictions = lrModel.transform(unlabeled_data)

predictions.show()

print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))