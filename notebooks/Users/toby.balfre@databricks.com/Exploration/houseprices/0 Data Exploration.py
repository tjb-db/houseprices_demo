# Databricks notebook source
# MAGIC %md # House Prices Regression Initial Data Exploration
# MAGIC 
# MAGIC Kaggle House Price Prediction competition - used for regression demo

# COMMAND ----------

# MAGIC %sql
# MAGIC USE houseprices;
# MAGIC SELECT current_database();

# COMMAND ----------

data = spark.sql("""SELECT * FROM houseprices.rawdata""")
display(data.take(5))

# COMMAND ----------

display(data.describe())

# COMMAND ----------

# MAGIC %md ### 1.1 MSSubClass Category Count

# COMMAND ----------

display(spark.sql("""SELECT count(*), MSSubClass FROM houseprices.rawdata GROUP BY MSSubClass"""))

# COMMAND ----------

# MAGIC %md ### 1.2 LotArea
# MAGIC Using analysis below, converting LotArea to buckets - 0, 4000, 8000, 12000, 18000

# COMMAND ----------

# MAGIC %md #### Histogram (Matplotlib)

# COMMAND ----------

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# create histogram using RDD Histogram function then plot
bins, counts = data.select('LotArea').rdd.flatMap(lambda x: x).histogram(100)

# specify graphs to show
ax.hist(bins[:-1], bins=bins, weights=counts)

# format the graph and set axis
plt.suptitle('Histogram Of Lot Size')
plt.xlabel('Lot Size In Square Feet')
plt.ylabel('Number Of Properties')

display(fig)

# COMMAND ----------

# MAGIC %md #### Histogram (Databricks Display - 1000 rows only)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT LotArea FROM rawdata

# COMMAND ----------

# MAGIC %md #### Bow & Whisper Plot (Databricks Display - 1000 rows only)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT LotArea FROM rawdata 