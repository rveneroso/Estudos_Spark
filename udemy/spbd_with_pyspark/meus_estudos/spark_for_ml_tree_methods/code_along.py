# Tree Methods Code Along

# Tree methods Example
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('treecode').getOrCreate()

# Load training data
data = spark.read.csv('College.csv', inferSchema=True, header=True)

data.printSchema()

data.head()

### Spark Formatting of Data

# A few things we need to do before Spark can accept the data!
# It needs to be in the form of two columns
# ("label","features")

# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

data.columns

assembler = VectorAssembler(
    inputCols=['Apps',
               'Accept',
               'Enroll',
               'Top10perc',
               'Top25perc',
               'F_Undergrad',
               'P_Undergrad',
               'Outstate',
               'Room_Board',
               'Books',
               'Personal',
               'PhD',
               'Terminal',
               'S_F_Ratio',
               'perc_alumni',
               'Expend',
               'Grad_Rate'],
    outputCol="features")

output = assembler.transform(data)

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="Private", outputCol="PrivateIndex")
output_fixed = indexer.fit(output).transform(output)

final_data = output_fixed.select("features", 'PrivateIndex')

train_data, test_data = final_data.randomSplit([0.7, 0.3])

### The Classifiers

from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier, RandomForestClassifier
from pyspark.ml import Pipeline

# Use mostly defaults to make this comparison "fair"

dtc = DecisionTreeClassifier(labelCol='PrivateIndex', featuresCol='features')
rfc = RandomForestClassifier(labelCol='PrivateIndex', featuresCol='features')
gbt = GBTClassifier(labelCol='PrivateIndex', featuresCol='features')

# Train the models (its three models, so it might take some time)
dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)

## Model Comparison


dtc_predictions = dtc_model.transform(test_data)
rfc_predictions = rfc_model.transform(test_data)
gbt_predictions = gbt_model.transform(test_data)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="PrivateIndex", predictionCol="prediction",
                                                  metricName="accuracy")

dtc_acc = acc_evaluator.evaluate(dtc_predictions)
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
gbt_acc = acc_evaluator.evaluate(gbt_predictions)

print("Here are the results!")
print('-' * 80)
print('A single decision tree had an accuracy of: {0:2.2f}%'.format(dtc_acc * 100))
print('-' * 80)
print('A random forest ensemble had an accuracy of: {0:2.2f}%'.format(rfc_acc * 100))
print('-' * 80)
print('A ensemble using GBT had an accuracy of: {0:2.2f}%'.format(gbt_acc * 100))
