from pyspark.sql import SparkSession
from pyspark.sql.functions import mean
spark = SparkSession.builder.appName('miss').getOrCreate()
df = spark.read.csv('ContainsNull.csv', inferSchema=True,header=True)
df.show()

# Apaga do DataFrame todas as linhas que contenham pelo menos uma coluna onde não exista valor
dn_without_missing_data = df.na.drop()
dn_without_missing_data.show()

# Apaga do DataFrame todas as linhas que contenham PELO MENOS 2 colunas onde haja valores.
dn_without_missing_data = df.na.drop(thresh=2)
dn_without_missing_data.show()

# Apaga do DataFrame todas as linhas nas quais todas as colunas sejam nulas. Como no DataFrame sendo utilizado não há nenhuma linha na qual todas as colunas estejam
# sem valor, o novo DataFrame será exatamente igual ao original.
dn_without_missing_data = df.na.drop(how='all')
dn_without_missing_data.show()

# Apaga do DataFrame todas as linhas nas quais não exista valor na coluna Sales. Mesmo que existam outras colunas que não possuam valor, somente serão
# excluídas as linhas nas quais a coluna Sales não contehha valor.
dn_without_missing_data = df.na.drop(subset=['Sales'])
dn_without_missing_data.show()

# Quando se utiliza na.fill, somente serão preenchidas as colunas cujo tipo seja aquele passado ao método na.fill.
# No exemplo abaixo é passada uma String. Existem valores faltantes na coluna Sales mas como ela é do tipo double então na.fill não será
# aplicada a ela. O resultado será:
# |  Id|           Name|Sales|
# +----+---------------+-----+
# |emp1|           John| null|
# |emp2|FILL WITH VALUE| null|
# |emp3|FILL WITH VALUE|345.0|
# |emp4|          Cindy|456.0|
# +----+---------------+-----+
dn_with_new_value = df.na.fill("FILL WITH VALUE")
dn_with_new_value.show()

# Obtém o valor médio da coluna Sales e usa esse valor para preencher as colunas do DataFrame nas quais não exista valor.
mean_sales = df.select(mean(df['Sales'])).collect()
mean_val = mean_sales[0][0]
dn_with_new_value = df.na.fill('No value',subset=['Name'])
dn_with_new_value = df.na.fill(mean_val,subset=['Sales'])
dn_with_new_value.show()