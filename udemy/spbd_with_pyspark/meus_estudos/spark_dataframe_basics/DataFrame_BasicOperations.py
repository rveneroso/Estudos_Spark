import pyspark
from pyspark.sql import SparkSession

# Cria uma sessão com o Spark
spark = SparkSession.builder.appName('ops').getOrCreate()

# Cria um DataFrame Spark a partir do arquivo appl_stock.csv. Arquivos csv oferecem mais opções de leitura do que arquivos json.
# A opção inferSchema permite que seja criado um schema para o DataFrame a partir do conteúdo do arquivo csv.
df = spark.read.csv('appl_stock.csv', inferSchema=True,header=True)

# Imprime o schema do DataFramew
df.printSchema()

# Filtra as linhas do DataFrame nas quais o valor close seja menor que 500
df.filter('Close < 500').show()

# Seleciona a coluna Open  das linhas do DataFrame nas quais o valor close seja menor que 500
df.filter('Close < 500').select('Open').show()

# Seleciona as colunas Open e Close das linhas do DataFrame nas quais o valor close seja menor que 500
df.filter('Close < 500').select(['Open','Close']).show()

# Filtra as linhas do DataFrame nas quais o valor close seja menor que 500. Aqui porém, são utilizados os comparadores padrão do Python.
df.filter(df['Close'] < 500).show()

# Para criar um filtro com múltiplas condições deve-se separar as condições por parênteses e utilizar os operadores & e | em vez de and e or.
# Seleciona as linhas nas quais o valor da coluna Close seja < 200 e o valor da coluna Open seja > 200
df.filter( (df['Close'] < 200) & (df['Open'] > 200) ).show()

# Seleciona as linhas nas quais o valor da coluna Close seja < 200 e o valor da coluna Open NÃO SEJA > 200
df.filter( (df['Close'] < 200) & ~(df['Open'] > 200) ).show()

# Selecionando um valor específico da coluna Low
df.filter(df['Low'] == 197.16).show()

# Capturando o resultado de um filter em vez de apenas exibí-lo.
result = df.filter(df['Low'] == 197.16).collect()
print(result)

# Recuperando a primeira linha da lista acima e convertendo-a para um dictionary
row = result[0]
my_dict = row.asDict()
print('--- Dictionary criado a partir de uma linha de um DataFrame ---')
print(my_dict)