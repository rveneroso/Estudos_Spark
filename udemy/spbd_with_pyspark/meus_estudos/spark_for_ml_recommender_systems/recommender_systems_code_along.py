# Consulting Project
## Recommender Systems - Solutions

### ATENÇÃO: ESSE CÓDIGO NÃO ESTÁ FUNCIONANDO E NO MOMENTO ESTOU SEM ÂNIMO PARA PROCURAR A SOLUÇÃO DO PROBLEMA ###

import pandas as pd

df = pd.read_csv('movielens_ratings.csv')

df.describe().transpose()

df.corr()

import numpy as np
df['mealskew'] = df['movieId'].apply(lambda id: np.nan if id > 31 else id)

df.describe().transpose()

mealmap = { 2. : "Chicken Curry",
           3. : "Spicy Chicken Nuggest",
           5. : "Hamburger",
           9. : "Taco Surprise",
           11. : "Meatloaf",
           12. : "Ceaser Salad",
           15. : "BBQ Ribs",
           17. : "Sushi Plate",
           19. : "Cheesesteak Sandwhich",
           21. : "Lasagna",
           23. : "Orange Chicken",
           26. : "Spicy Beef Plate",
           27. : "Salmon with Mashed Potatoes",
           28. : "Penne Tomatoe Pasta",
           29. : "Pork Sliders",
           30. : "Vietnamese Sandwich",
           31. : "Chicken Wrap",
           np.nan: "Cowboy Burger",
           4. : "Pretzels and Cheese Plate",
           6. : "Spicy Pork Sliders",
           13. : "Mandarin Chicken PLate",
           14. : "Kung Pao Chicken",
           16. : "Fried Rice Plate",
           8. : "Chicken Chow Mein",
           10. : "Roasted Eggplant ",
           18. : "Pepperoni Pizza",
           22. : "Pulled Pork Plate",
           0. : "Cheese Pizza",
           1. : "Burrito",
           7. : "Nachos",
           24. : "Chili",
           20. : "Southwest Salad",
           25.: "Roast Beef Sandwich"}

df['meal_name'] = df['mealskew'].map(mealmap)

df.to_csv('Meal_Info.csv',index=False)

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('recconsulting').getOrCreate()

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

data = spark.read.csv('Meal_Info.csv',inferSchema=True,header=True)

(training, test) = data.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="mealskew", ratingCol="rating")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)

predictions.show()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))