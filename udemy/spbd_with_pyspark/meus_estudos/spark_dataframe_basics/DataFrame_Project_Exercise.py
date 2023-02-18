from pyspark.sql import SparkSession, functions as F

# Start a simple Spark Session
from pyspark.sql.functions import format_number

spark = SparkSession.builder.appName('DFProjectExercise').getOrCreate()

# Load the Walmart Stock CSV File, have Spark infer the data types.
df = spark.read.csv('walmart_stock.csv',header=True,inferSchema=True)

# What are the column names?
print(df.columns)

# What does the Schema look like?
df.printSchema()

# Print out the first 5 columns (actually, rows).
print(df.head(5))
# or
for row in df.head(5):
    print(row)
    print('\n')

# Use describe() to learn about the DataFrame.
df.describe().show()

# Bonus question
result = df.describe()
result.select(result['summary'],
              format_number(result['Open'].cast('float'),2).alias('Open'),
              format_number(result['High'].cast('float'),2).alias('High'),
              format_number(result['Low'].cast('float'),2).alias('Low'),
              format_number(result['Close'].cast('float'),2).alias('Close'),
              result['Volume'].cast('int').alias('Volume')
             ).show()

# Create a new dataframe with a column called HV Ratio that is the ratio of the High Price versus volume of stock traded for a day.
new_df = df.withColumn("Ratio",df["High"]/df["Volume"])
new_df.select("Ratio").show()

# What day had the Peak High in Price?
print(new_df.orderBy(new_df["High"].desc()).head(1)[0][0])

# What is the mean of the Close column?
new_df.select(F.mean(new_df.Close)).show()

# What is the max and min of the Volume column?
new_df.select(F.max(new_df.Volume), F.min(new_df.Volume)).show()

# How many days was the Close lower than 60 dollars?
print(new_df.filter("Close < 60").count())
# or
print(new_df.filter(df['Close'] < 60).count())

# What percentage of the time was the High greater than 80 dollars ?
count = new_df.filter(df["High"] > 80).count();
print((count / new_df.count()) * 100)

# What is the Pearson correlation between High and Volume?
new_df.select(F.corr("High","Volume")).show()

# What is the max High per year?
new_df = df.withColumn('Year',F.year(df['date']))
print(new_df.groupby("Year").agg(F.max('High')).collect())
# or
yeardf = df.withColumn("Year",F.year(df["Date"]))
max_df = yeardf.groupBy('Year').max()
max_df.select('Year','max(High)').show()

# What is the average Close for each Calendar Month?
new_df = df.withColumn('Month',F.month(df['date']))
print(new_df.groupby("Month").agg(F.mean('Close')).collect())
# or
monthdf = df.withColumn("Month",F.month("Date"))
monthavgs = monthdf.select("Month","Close").groupBy("Month").mean()
monthavgs.select("Month","avg(Close)").orderBy('Month').show()