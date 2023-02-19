from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('lr_example').getOrCreate()
from pyspark.ml.regression import LinearRegression
# Load training data
training = spark.read.format("libsvm").load("sample_linear_regression_data.txt")
training.show()
# These are the default values for the featuresCol, labelCol, predictionCol
lr = LinearRegression(featuresCol='features', labelCol='label', predictionCol='prediction')

# You could also pass in additional parameters for regularization, do the reading
# in ISLR to fully understand that, after that its just some simple parameter calls.
# Check the documentation with Shift+Tab for more info!
# Fit the model
lrModel = lr.fit(training)
# Print the coefficients and intercept for linear regression
print("Coefficients: {}".format(str(lrModel.coefficients))) # For each feature...
print('\n')
print("Intercept:{}".format(str(lrModel.intercept)))
# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary

trainingSummary.residuals.show()
print("RMSE: {}".format(trainingSummary.rootMeanSquaredError))
print("r2: {}".format(trainingSummary.r2))

all_data = spark.read.format("libsvm").load("sample_linear_regression_data.txt")
# No correct, but generally 70/30 or 60/40 splits are used.
# Depending on how much data you have and how unbalanced it is.
train_data,test_data = all_data.randomSplit([0.7,0.3])
train_data.show()
test_data.show()

unlabeled_data = test_data.select('features')
unlabeled_data.show()

correct_model = lr.fit(train_data)
test_results = correct_model.evaluate(test_data)
test_results.residuals.show()
print("RMSE: {}".format(test_results.rootMeanSquaredError))

predictions = correct_model.transform(unlabeled_data)
predictions.show()