# Clustering Documentation Example

#Cluster methods Example
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('cluster').getOrCreate()

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Loads data.
dataset = spark.read.format("libsvm").load("sample_kmeans_data.txt")

# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
# ATENÇÃO!!! O método computeCost não existe mais na versão do pyspark que estou usando. Para descobrir o wssse agora é preciso utilizar o ClusteringEvaluator.
# wssse = model.computeCost(dataset)
predictions = model.transform(dataset)
# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

wssse = evaluator.evaluate(predictions)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

