#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import RankingMetrics

def main(spark, model_formulation, data_file, val_file, model_file):
    df = spark.read.parquet(data_file).sample(0.001)
    df_val = spark.read.parquet(val_file).sample(0.001)
    
    #Extension 1: alternative model formulations
    
    if model_formulation == 'log':
        #log compression on training
        df = df.withColumn('count', log(col('count')))
   
        #log compression on validation
        df_val = df_val.withColumn('count', log(col('count')))

    elif model_formulation == 'ct1':
        #subsetting all train counts greater than 1
        df.createOrReplaceTempView('df')
        df = spark.sql('SELECT * FROM df WHERE count > 1')

        #subsetting all val counts greater than 1
        df_val.createOrReplaceTempView('df_val')
        df_val = spark.sql('SELECT * FROM df_val WHERE count > 1')
    
    elif model_formulation == 'ct2':
        #subsetting all train counts greater than 2
        df.createOrReplaceTempView('df')
        df = spark.sql('SELECT * FROM df WHERE count > 2')
        
        #subsetting all val counts greater than 2
        df_val.createOrReplaceTempView('df_val')
        df_val = spark.sql('SELECT * FROM df_val WHERE count > 2')

    uid_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_num",\
                        handleInvalid='keep')
    tid_indexer = StringIndexer(inputCol="track_id", outputCol="track_id_num",\
                        handleInvalid='keep')

    pipeline = Pipeline(stages=[uid_indexer, tid_indexer])
    map_model = pipeline.fit(df)
    val_map_model = pipeline.fit(df_val)
    mapping = map_model.transform(df)
    df_val = val_map_model.transform(df_val)
    
    #perform cross validation on these ranges
    regParams = [0.001, 0.01]#,  0.1]
    alphas = [0.8, 1]#, 1.2]
    ranks = [8, 10]#, 12]
    
    #dictionary to hold best parameters (always)
    best_params = {}
    
    #initialize dict w/ params 
    best_params['regParam']=regParams[0]
    best_params['alpha']=alphas[0]
    best_params['rank']=ranks[0]
    curr_MAP = 100 #random max initialization so MAP < curr_MAP is true

    for p in regParams:
        for a in alphas:
            for r in ranks:
                als = ALS(maxIter=5, regParam=p, alpha=a,rank=r,\
                            userCol="user_id_num", itemCol="track_id_num",\
                            ratingCol="count", implicitPrefs=True)

                #fit on training + check on validation
                als_model = als.fit(mapping)
                
                #create actual items dataframe
                actual_recs = df_val.groupBy('user_id_num')\
                                .agg(F.collect_list('track_id_num')
                                .alias('track_id_num'))

                #create user predicted items dataframe
                user_subset = df_val.select('user_id_num').distinct()
                pred_recs = als_model.recommendForUserSubset(user_subset,10)
                pred_recs = pred_recs.select('user_id_num',\
                                col('recommendations.track_id_num')\
                                .alias('track_id_num'))

                #create RDD & join on users
                perUserItemsRDD = pred_recs\
                                    .join(actual_recs, on='user_id_num').rdd\
                                    .map(lambda row: (row[1],row[2]))

                rankingMetrics = RankingMetrics(perUserItemsRDD)
                MAP = rankingMetrics.meanAveragePrecision

                #keep track of best parameters that minimize rmse
                if MAP < curr_MAP:
                    best_params['regParam']=p
                    best_params['alpha']=a
                    best_params['rank']=r
    
    print("Best regParam: ",best_params['regParam'])
    print("Best alpha: ",best_params['alpha'])
    print("Best rank: ",best_params['rank'])

    #use these best parameters to fit an ALS model
    best_als_model = ALS(maxIter=5, regParam=best_params['regParam'],\
                alpha=best_params['alpha'],rank=best_params['rank'],\
                implicitPrefs=True,userCol="user_id_num",\
                itemCol="track_id_num")
    #save best model
    best_als_model.write().overwrite().save(model_file)
                

# Only enter this block if we're in main
if __name__ == "__main__":

    spark = SparkSession.builder.appName('ext_val_als').getOrCreate()
    model_formulation = sys.argv[1]
    data_file = sys.argv[2]
    val_file = sys.argv[3]
    model_file = sys.argv[4]

    main(spark, model_formulation, data_file, val_file, model_file)
