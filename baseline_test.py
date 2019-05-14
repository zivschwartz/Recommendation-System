#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexer
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col

def main(spark, model_file, data_file):
   
    #load ALS model
    als_model = ALSModel.load(model_file)

    #read in test data as parquet
    df_test = spark.read.parquet(data_file)
    uid_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx",\
                        handleInvalid='keep')
    tid_indexer = StringIndexer(inputCol="track_id", outputCol="track_idx",\
                        handleInvalid='keep')

    pipeline = Pipeline(stages=[uid_indexer, tid_indexer])
    map_model = pipeline.fit(df_test)
    df_test = map_model.transform(df_test)

    ########### PERFORM RANKING METRICS ##########
    #create user actual items dataframe
    actual_recs = df_test.groupBy('user_idx')\
                    .agg(F.collect_list('track_idx').alias('track_idx'))
     
    #create user predicted items dataframe
    user_subset = df_test.select('user_idx').distinct()
    pred_recs = als_model.recommendForUserSubset(user_subset,500)
    pred_recs = pred_recs.select('user_idx',\
                    col('recommendations.track_idx').alias('track_idx'))

    #create user item RDD & join on users 
    perUserItemsRDD = pred_recs\
                        .join(actual_recs, on='user_idx').rdd\
                        .map(lambda row: (row[1], row[2]))
    
    rankingMetrics = RankingMetrics(perUserItemsRDD)
    #print results to the console
    print("Ranking Metrics MAP: {}",rankingMetrics.meanAveragePrecision)



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline_test').getOrCreate()

    # And the location to store the trained model
    model_file = sys.argv[1]

    # Get the filename from the command line
    data_file = sys.argv[2]

    # Call our main routine
    main(spark, model_file, data_file)
