# Databricks notebook source
# MAGIC %md # House Prices Regression Feature Engineering Pipeline
# MAGIC 
# MAGIC Kaggle House Price Prediction competition - used for regression demo

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer, RFormula, VectorAssembler, Bucketizer, QuantileDiscretizer, MinMaxScaler, OneHotEncoderEstimator, StringIndexer

# COMMAND ----------

# MAGIC %md ##1. Database Select and Data Load

# COMMAND ----------

# MAGIC %sql
# MAGIC USE houseprices;
# MAGIC SELECT current_database();

# COMMAND ----------

data = spark.sql("""SELECT * FROM houseprices.rawdata""")
display(data.take(5))

# COMMAND ----------

data.columns

# COMMAND ----------

# MAGIC %md ##2. Feature Engineering

# COMMAND ----------

# subset data columns from raw data to convert into features. 
# cast datatypes and rename
columnSubset = SQLTransformer()\
  .setStatement("""
  SELECT
  MSSubClass
  , LotArea
  , MSZoning
  , GrLivArea
  , SalePrice as label
  FROM __THIS__
  """)

# COMMAND ----------

# MAGIC %md ### 2.1 Dwelling Type Feature Prep

# COMMAND ----------

# convert categorical variables into string indexes ready for conversion to binary values using onehotencoder
msSubStringIndexer = StringIndexer()\
  .setInputCol('MSSubClass')\
  .setOutputCol('MSSubIndexed')\
  .setStringOrderType('frequencyDesc')\
  .setHandleInvalid('keep')

# COMMAND ----------

# MAGIC %md ### 2.2 LotArea Convert To Bins

# COMMAND ----------

# convert LotArea continous variable to bins. Analysis showed to be highly skewed to multiple outliers representing high value houses
laBucketizer = Bucketizer()\
  .setSplits([0, 4000, 8000, 12000, 18000, float("inf")])\
  .setInputCol('LotArea')\
  .setOutputCol('LABuckets')\
  .setHandleInvalid('skip')

msSubStringIndexer = StringIndexer()\
  .setInputCol('LABuckets')\
  .setOutputCol('LAIndexed')\
  .setStringOrderType('frequencyDesc')\
  .setHandleInvalid('skip')

# COMMAND ----------

# MAGIC %md ### 2.3 MSZoning Binarizer

# COMMAND ----------

msZoningIndexer = StringIndexer()\
  .setInputCol('MSZoning')\
  .setOutputCol('MSZoningIndexed')\
  .setStringOrderType('frequencyDesc')\
  .setHandleInvalid('skip')

# COMMAND ----------

# MAGIC %md ### 2.4 GrLivArea Rank Feature

# COMMAND ----------

# use sql to generate percentiles and link back to the percentile for each property
LivRank = SQLTransformer()\
  .setStatement("""
    SELECT *
    , rank(GrLivArea) OVER (PARTITION BY MSZoning ORDER BY GrLivArea DESC) AS arearank
    FROM __THIS__
  """)

# COMMAND ----------

# MAGIC %md ### 2.5 SubClass Join In New Variable 

# COMMAND ----------

subClassJoin = SQLTransformer()\
  .setStatement("""
    SELECT a.*
    , b.FamilyInterest as faminterest
    FROM __THIS__ a
    LEFT JOIN houseprices.subconversion b
    ON a.MSSubClass = b.msSubClass
  """)

# COMMAND ----------

subClassIndexer = StringIndexer()\
  .setInputCol('faminterest')\
  .setOutputCol('famIndexed')\
  .setStringOrderType('frequencyDesc')\
  .setHandleInvalid('skip')

# COMMAND ----------

# MAGIC %md ###2.6 Create Binary Categorical Variables

# COMMAND ----------

# convert categorical variables into binary variables (requires string indexer to be run first)
encoder = OneHotEncoderEstimator()\
  .setInputCols(['MSSubClass', 'LAIndexed', 'MSZoningIndexed', 'famIndexed'])\
  .setOutputCols(['msSubCats', 'LACats', 'ZoningCats', 'famCats'])\
  .setHandleInvalid('keep')\
  .setDropLast(False)

# COMMAND ----------

# MAGIC %md ### 2.7 Feature Vector Build and Scaling

# COMMAND ----------

toInclude = ['msSubCats', 'LACats', 'ZoningCats', 'famCats']
assembler = VectorAssembler()\
  .setInputCols(toInclude)\
  .setOutputCol('features')

# COMMAND ----------

# MAGIC %md ## 3. Pipeline Configure and Run

# COMMAND ----------

stages = [columnSubset, laBucketizer, msSubStringIndexer, msZoningIndexer, LivRank, subClassJoin, subClassIndexer, encoder, assembler]
pipeline = Pipeline().setStages(stages)

# COMMAND ----------

training = pipeline.fit(data).transform(data)
training.registerTempTable("temptraining")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- persist table to database for consumption by model building pipeline
# MAGIC DROP TABLE IF EXISTS houseprices.training;
# MAGIC CREATE TABLE houseprices.training AS
# MAGIC SELECT *
# MAGIC FROM temptraining

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM houseprices.training where MSZoning = 'RH' limit 50