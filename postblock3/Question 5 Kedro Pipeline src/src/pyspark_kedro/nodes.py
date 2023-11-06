"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

# Import necessary libraries for the functions
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests
from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler, Imputer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("Kedro-Pyspark-Project").getOrCreate()

def read_data():
    internal_data_url = "https://storage.googleapis.com/bdt-spark-store/internal_data.csv"
    external_sources_url = "https://storage.googleapis.com/bdt-spark-store/external_sources.csv"

    internal_file = "data/01_raw/gcs_internal_data.csv"
    external_file = "data/01_raw/gcs_external_sources.csv"

    response = requests.get(internal_data_url)
    with open(internal_file, "wb") as file:
        file.write(response.content)

    response = requests.get(external_sources_url)
    with open(external_file, "wb") as file:
        file.write(response.content)

    df_internal_data = spark.read.csv('data/01_raw/gcs_internal_data.csv', header=True, inferSchema=True)
    df_external_data  = spark.read.csv('data/01_raw/gcs_external_sources.csv', header=True, inferSchema=True)
    return [df_internal_data, df_external_data]

# Function to join datasets
def join_data(df_internal_data: DataFrame, df_external_data: DataFrame)  -> DataFrame:
    df_join = df_internal_data.join(df_external_data, "SK_ID_CURR", "inner")

    columns_extract = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                  'DAYS_BIRTH', 'DAYS_EMPLOYED', 'NAME_EDUCATION_TYPE',
                  'DAYS_ID_PUBLISH', 'CODE_GENDER', 'AMT_ANNUITY',
                  'DAYS_REGISTRATION', 'AMT_GOODS_PRICE', 'AMT_CREDIT',
                  'ORGANIZATION_TYPE', 'DAYS_LAST_PHONE_CHANGE',
                  'NAME_INCOME_TYPE', 'AMT_INCOME_TOTAL', 'OWN_CAR_AGE', 'TARGET']  

    df_select = df_join.select(*columns_extract)
    df_select.show(4)    
    return df_select

# Function to split data
def split_data(df_select: DataFrame)  -> Tuple:
    train, test = df_select.randomSplit([0.8, 0.2], seed=101)
    return train, test

# Function to preprocess data
def preprocess_data(train: DataFrame, test: DataFrame)  -> Tuple:
    stages = []
    numericCols = [col for col, dtype in train.dtypes if dtype in ["double", "int"]]
    imputer = Imputer(inputCols=numericCols, outputCols=[f"{c}_imputed" for c in numericCols])
    stages += [imputer]
    imputedNumericCols = [f"{c}_imputed" for c in numericCols]

    categoricalColumns = [col for col, dtype in train.dtypes if dtype == "string"]
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "_Index")
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "_classVec"])
        stages += [stringIndexer, encoder]

    assemblerInputs = [c + "_classVec" for c in categoricalColumns] + imputedNumericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    stages += [scaler]

    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(train)
    train_data = pipelineModel.transform(train)
    test_data = pipelineModel.transform(test)

    return train_data, test_data, assemblerInputs

# Function to train the model
def train_model(train_data: DataFrame)  -> RandomForestClassifier:
    rfClassifier = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="TARGET", numTrees=100, seed=50)
    return rfClassifier.fit(train_data)

# Function to predict
def predict(model: RandomForestClassifier, test_data: DataFrame)  -> DataFrame:
    predictions = model.transform(test_data)
    return predictions

# Function to evaluate the model
def evaluate_model(predictions: DataFrame) -> None:
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="TARGET", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="TARGET", predictionCol="prediction", metricName="accuracy")
    accuracy = accuracy_evaluator.evaluate(predictions)
    print(f"Accuracy: {accuracy:.4f} AUC: {auc:.4f}")
    print(" ")

# Function to extract feature importances
def feature_importance(model: RandomForestClassifier, assemblerInputs: DataFrame) -> None:
    importances = model.featureImportances
    importance_list = [(assemblerInputs[i], importances[i]) for i in range(len(assemblerInputs))]
    sorted_importances = sorted(importance_list, key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importances[:10]:
         print(f"Feature: {feature}, Importance: {importance:.6f}")
    print(" ")
