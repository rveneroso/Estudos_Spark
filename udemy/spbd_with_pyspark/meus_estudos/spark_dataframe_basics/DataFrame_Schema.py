from pyspark.sql.types import StructField,StringType,IntegerType,StructType
from pyspark.sql import SparkSession

# Cria uma sessão com o Spark
spark = SparkSession.builder.appName('Basics').getOrCreate()

# Cria uma variável que define a estrutura dos campos age e name
data_schema = [StructField("age", IntegerType(), True),StructField("name", StringType(), True)]

# Cria uma variável que define uma estrutura de dados baseada na variável data_schema criada acima.
final_struc = StructType(fields=data_schema)

# Cria um DataFrame a partir do arquivo people.json indicando que o schema do DataFrame é aquele definido pela variável final_struc.
df = spark.read.json('people.json', schema=final_struc)

# Exibe o schema do DataFrame
df.printSchema()