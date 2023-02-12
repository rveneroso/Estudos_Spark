import pyspark
from pyspark.sql import SparkSession

# Cria uma sessão com o Spark
spark = SparkSession.builder.appName('Basics').getOrCreate()

# Cria um DataFrame Spark a partir do arquivo people.json
df = spark.read.json('people.json')

# Exibe o conteúdo do DataFrame em forma de tabela
df.show()

# Exibe o schema do DataFrame
df.printSchema()

# Exibe o nome das colunas do DataFrame. Lembrando que a forma de executar o print mudou no Python 3.
print(df.columns)

# Descreve o DataFrame no que diz respeito às suas métricas (número de elementos, média, desvio padrão etc.)
df.describe().show()

# O método select aplicado em um DataFrame retorna um outro DataFrame. No exemplo abaixo, o print exibirá: <class 'pyspark.sql.dataframe.DataFrame'>
age_df = df.select('age')
print(type(age_df))

# Para visualizar os valores do DataFrame criado acima basta utilizar a função show()
age_df.show()

# Visualizando as duas primeiras linhas de um DataFrame. Lembrando que o print no Python 3 não é mais implícito como era nas versões anteriores.
# Antes do Python 3 bastava executar df.head(2) que as duas primeiras linhas do DataFrame eram exibidas. Agora é necessário aplicar print()
# para visualizar as mesmas linhas.
print(df.head(2))

# É possível selecionar mais de uma coluna de um DataFrame na mesma operação. Observar que, nesse caso é preciso passar uma lista contendo os nomes
# das colunas a serem retornadas.
age_name_df = df.select(['age','name'])
print(type(age_name_df))

# Criando uma coluna em um DataFrame. No exemplo abaixo, a coluna newage é criada e recebe os valores já existentes na coluna age.
# Observação: se a aplicação do método withColumn não for atribibuída à um novo DataFrame, o DataFrame original NÃO SERÁ ALTERADO.
# Na linha abaixo, se não houve a atribuição à variável new_df, o DataFrame df não seria alterado.
new_df = df.withColumn('newage', df['age'])
new_df.show()

# Renomeando uma coluna do DataFramew. Novamente: o método withColumnRename não altera o DataFrame original. Por isso, na linha abaixo, o resultado é atribuído
# á new_df.
new_df = new_df.withColumnRenamed('newage','new_age')
new_df.show()

# Registrando uma temporary view. A linha abaixo cria um temporary view a partir do DataFrame df e a torna visível para uso posterior dentro do código.
df.createOrReplaceTempView('people')

# Selecionando todos os registros da temporary view people
print('--- Exibindo o conteúdo da temporary view people ---')
results = spark.sql("SELECT * FROM people")
results.show()

# Aplicando condições a uma temporary view.
new_results = spark.sql(("SELECT * FROM people WHERE age = 30"))
new_results.show()