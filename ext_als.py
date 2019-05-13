#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F 

def main(spark, data_file, model_file, user_file, track_file, model_formulation = None):
    df = spark.read.parquet(data_file)
   
    if model_formulation == 'log':
        #log compression on training
        df = df.withColumn('count', log(F.col('count')))
   
    elif model_formulation == 'ct1':
        #subsetting all train counts greater than 1
        df.createOrReplaceTempView('df')
        df = spark.sql('SELECT * FROM df WHERE count > 1')

    elif model_formulation == 'ct2':
        #subsetting all train counts greater than 2
        df.createOrReplaceTempView('df')
        df = spark.sql('SELECT * FROM df WHERE count > 2')
    
    else:
        #If no model formulation is specified, continue
        continue

    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx", handleInvalid="keep")
    track_indexer = StringIndexer(inputCol="track_id", outputCol="track_idx", handleInvalid="keep")

    pipeline = Pipeline(stages=[user_indexer, track_indexer])
    mapping = pipeline.fit(df)
    df = mapping.transform(df)
    
    #create + fit an ALS model
    als = ALS(maxIter=5, regParam=0.01, implicitPrefs=True, ratingCol="count", userCol="user_idx", itemCol="track_idx")
    als_model = als.fit(df)
    
    #save trained ALS model
    als_model.write().overwrite().save(model_file)
    print("Model sucessfully saved to HFS")
    
    #save string indexers
    user_indexer.write().overwrite().save(user_file)
    track_indexer.write().overwrite().save(track_file)
    print("String Indexers sucessfully saved to HFS")

# Only enter this block if we're in main
if __name__ == "__main__":

    spark = SparkSession.builder.appName('ext_als').getOrCreate()
    data_file = sys.argv[1]
    model_file = sys.argv[2]
    user_file = sys.argv[3]
    track_file = sys.argv[4]
    model_formulation = sys.argv[5]

    main(spark, data_file, model_file, user_file, track_file, model_formulation)
