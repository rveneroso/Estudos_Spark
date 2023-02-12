from pyspark.sql import SparkSession
from pyspark.sql.functions import (dayofmonth,hour,month,
                                   dayofyear,year,weekofyear,
                                   format_number, date_format)

spark = SparkSession.builder.appName('dates').getOrCreate()

df = spark.read.csv('appl_stock.csv', inferSchema=True,header=True)

# Extraindo o dia do mês da data
df.select(dayofmonth(df['date'])).show()

# Extraindo a hora da data
df.select(hour(df['date'])).show()

# Extraindo o mês da data
df.select(month(df['date'])).show()

# Cria um novo DataFrame a partir de df com a adição da coluna Year
newdf = df.withColumn('Year',year(df['date']))

# Obtém a média por ano do valor da coluna Close
result = newdf.groupBy('Year').mean().select(['Year','avg(Close)'])
# result.withColumnRenamed('avg(Close)','Average Closing Price per Year').show()
result = result.select('Year',format_number('avg(Close)',2).alias("Mean Close")).show()