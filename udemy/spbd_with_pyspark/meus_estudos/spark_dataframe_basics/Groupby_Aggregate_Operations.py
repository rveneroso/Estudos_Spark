from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('aggs').getOrCreate()
df = spark.read.csv('sales_info.csv', inferSchema=True,header=True)

# Agrupa os dados pela coluna Company e exibe a média de cada grupo
df.groupby("Company").mean().show()

# Soma os valores da coluna Sales no DataFrame utilizando um dicionário que define a coluna e a operação a ser realizada na agregação.
df.agg({'Sales':'sum'}).show()

# Obtendo o valor máximo do atributo Sales dentro de cada Company.
group_data = df.groupby('Company')
group_data.agg({'Sales':'max'}).show()

# Importando mais funções do pyspark
from pyspark.sql.functions import countDistinct,avg,stddev,format_number

# Calcula o valor médio de vendas (Sales) usando a função importada acima e a exibe com o cabeçalho 'Average Sales'
df.agg(avg('Sales').alias('Average Sales')).show()

sales_std = df.select(stddev("Sales").alias('std'))
sales_std.select(format_number('std',2).alias('std')).show()

# Ordenando o DataFrame pela coluna Sales
df.orderBy('Sales').show()

# Para ordenar em ordem decrescente é preciso informar a coluna propriamente dita e não apenas o seu nome.
df.orderBy(df['Sales'].desc()).show()
