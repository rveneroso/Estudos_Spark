from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('logregdoc').getOrCreate()

from pyspark.ml.classification import LogisticRegression

# Load training data
training = spark.read.format("libsvm").load("sample_libsvm_data.txt")

lr = LogisticRegression()

# Fit the model
lrModel = lr.fit(training)

trainingSummary = lrModel.summary

trainingSummary.predictions.show()
# May change soon!
from pyspark.mllib.evaluation import MulticlassMetrics
lrModel.evaluate(training)

# Usually would do this on a separate test set!
predictionAndLabels = lrModel.evaluate(training)

predictionAndLabels.predictions.show()

predictionAndLabels = predictionAndLabels.predictions.select('label','prediction')

predictionAndLabels.show()

from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator

evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='label')

# For multiclass
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label',
                                             metricName='accuracy')

acc = evaluator.evaluate(predictionAndLabels)