"""
Big Data Analytics Techniques and Applications 
Final Project
Author: Ming-Yu Lin
"""

# %%
# import Spark MLlib and Spark functions
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.linalg import SparseVector, DenseVector
from tqdm import tqdm
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# %%
def dataloader(path, window=10, macro=False, stock=True, technical_indicator=False):
    # get all files in path
    files = os.listdir(path)
    datas = []
    labels = []
    for file in tqdm(files):
        # read csv file
        df = pd.read_csv(f"{path}/{file}")
        if file == "df_AMGN.csv": # aviod broken file, AMGN
            continue
        # drop useless columns
        df = df.drop(["date","tic"], axis=1)
        # calculate close price pct change and label according to pct change
        df["change"] = df["close"].pct_change()
        df["label"] = df["change"].apply(lambda x: 1 if x > 0 else 0)
        # setup technical indicator
        if technical_indicator == False:
            df = df.drop(["volume","day","macd","boll_ub",
                          "boll_lb","rsi_30","cci_30","dx_30",
                          "close_30_sma","close_60_sma"], axis=1)
        # setup macro
        if macro == False:
            df = df.drop(["EFFR","DTB3","T10YIE","DGS10","GC=F","CL=F"], axis=1)
        # for each window, we use the data of previous window days to predict the label of next day
        for i in range(1, len(df), window):
            if i+window-1 >= len(df):
                break
            else:
                data = df.iloc[i:i+window-1, :-1].to_numpy().T # transpose for better utilization
                data = data.reshape(1, -1) # (1, 189)
                label = df.iloc[i+window-1, -1].reshape(-1, 1) # (1, 1)
                datas.append(data)
                labels.append(label)
            
    datas = np.concatenate(datas, axis=0)
    print(f"Data shape is {datas.shape}")
    labels = np.concatenate(labels, axis=0)
    print(f"Label shape is {labels.shape}")
    # seperate train and test data
    train_data = datas[:int(len(datas)*0.8)]
    train_label = labels[:int(len(labels)*0.8)]
    train_dataset = np.concatenate((train_data, train_label), axis=1)
    print(f"Train dataset shape is {train_dataset.shape}")
    train_dataset = pd.DataFrame(train_dataset, columns=[f"f{i}" for i in range(len(train_dataset[0]) - 1)]+["label"])
    test_data = datas[int(len(datas)*0.8):]
    test_label = labels[int(len(labels)*0.8):]
    test_dataset = np.concatenate((test_data, test_label), axis=1)
    print(f"Test dataset shape is {test_dataset.shape}")
    test_dataset = pd.DataFrame(test_dataset, columns=[f"f{i}" for i in range(len(test_dataset[0]) - 1)]+["label"])
    return train_dataset, test_dataset

# %%
def model_selector(model_name="LogisticRegression"):
    """
    Select model
    """
    if model_name == "LogisticRegression":
        return LogisticRegression(labelCol="label",featuresCol="features")
    elif model_name == "RandomForestClassifier":
        return RandomForestClassifier(labelCol="label",featuresCol="features", numTrees=10)
    elif model_name == "GBTClassifier":
        return GBTClassifier(labelCol="label",featuresCol="features", maxIter=10)
    elif model_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(labelCol="label",featuresCol="features")
    else:
        raise ValueError("model_name is not defined")

# %%
def fit_model(df_train, model_name="LogisticRegression"):
    """
    Build logistic regression pipeline
    return: pipelineModel
    """
    # Build pipeline
    # 1. VectorAssembler (Conat each feature into a vector which named "features")
    features = df_train.columns[:-1]
    assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
    # 2. Model selection
    model = model_selector(model_name)
    # 3. Pipeline
    pipeline = Pipeline(stages=[assembler, model])
    # 4. fit pipeline
    pipelineModel = pipeline.fit(df_train)
    return pipelineModel

# %%
def evaluate_model(pipelineModel, df_test):
    """
    Evaluate model
    """
    # 5. predict by training pipelineModel
    predicted = pipelineModel.transform(df_test)
    predicted.select("features", "label", "prediction").show(5)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", 
                                                predictionCol="prediction", 
                                                metricName="accuracy")
    accuracy = evaluator.evaluate(predicted)
    return accuracy
        
def get_feature_importance(window=20):
    train_dataset, test_dataset = dataloader("./data", window=window, macro=True, stock=True, technical_indicator=True)
    train_dataset = spark.createDataFrame(train_dataset)
    test_dataset = spark.createDataFrame(test_dataset)
    features = train_dataset.columns[:-1]
    assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
    train_dataset = assembler.transform(train_dataset)
    rf = GBTClassifier(labelCol="label",featuresCol="features", maxIter=10)
    model = rf.fit(train_dataset)
    feature_importance = model.featureImportances
    test_arr = feature_importance.toArray()
    new_score = []
    for i in range(0, len(test_arr), window-1):
        new_score.append(np.sum(test_arr[i:i+window-1]))
    df = pd.read_csv("./data/df_WBA.csv")
    importance_df = pd.DataFrame(np.array(new_score[:len(new_score) - 1]).reshape(1, window_size), columns = df.columns[2:])
    return importance_df

# %%
if __name__ == "__main__":
    # Set SparkConfig and Start SparkSession
    conf = SparkConf().setAppName("FinalProject").setMaster("local[24]") \
                    .set("spark.executor.memory", "16g").set("spark.driver.memory", "16g")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    
    # models
    model_names = ["LogisticRegression", "RandomForestClassifier", "GBTClassifier", 
                "DecisionTreeClassifier"]
    
    # macro, stock technical indicator
    combinations = [[True, True, True], # M + S + T
                    [True, True, False], # M + S
                    [False, True, True], # S + T
                    [False, True, False] ] # S
    combinations_name = ["M + S + T", "M + S", "S + T", "S"]
    window_size = [5, 10, 20]

    # test different combinations
    accuracys = []
    names = []
    for model in model_names: 
        for window in window_size: 
            for combination in combinations:
                macro, stock, technical_indicator = combination
                print(f"Model: {model}, window: {window}, combination: {combination}")
                train_dataset, test_dataset = dataloader("./data", window=window, macro=macro, stock=stock, technical_indicator=technical_indicator)
                train_dataset = spark.createDataFrame(train_dataset)
                test_dataset = spark.createDataFrame(test_dataset)
                pipelineModel = fit_model(train_dataset, model_name=model)
                accuracy = evaluate_model(pipelineModel, test_dataset)
                accuracys.append(accuracy)
                print(f"{model} accuracy is {accuracy}, window is {window}, combination is {combination}")

    # store results to csv
    accuracys = np.array(accuracys).reshape(4, 3, 4)
    for i in range(3):
        df = pd.DataFrame(accuracys[:, i, :], index=model_names, columns=combinations_name)
        df.to_csv(f"./results/Window_{window_size[i]}.csv", sep="\t")
    
    get_feature_importance(window=20)