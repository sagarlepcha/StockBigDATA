import pandas as pd
import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Set environment variables for Spark and Java
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("StockBetaCalculation") \
    .config("spark.executor.memory", "8g") \
    .config("spark.network.timeout", "600s") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Load the dataset
data_path = "hdfs://namenode:8020/input/TCS_with_indicators_.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Convert 'date' column to timestamp
df = df.withColumn("date", col("date").cast("timestamp"))

# Sort data by date
df = df.orderBy("date")

# Drop rows with null values in relevant columns
df = df.dropna(subset=["open", "high", "low", "close", "volume", "BETA"])

# Select the features and the target column (BETA)
feature_columns = ["open", "high", "low", "close", "volume"]

# Assemble features into a vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df = assembler.transform(df)

# Split data into training and test sets (80/20 split)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Initialize the linear regression model
lr = LinearRegression(featuresCol="features", labelCol="BETA")

# Fit the model on the training data
lr_model = lr.fit(train_data)

# Print model coefficients and intercept
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

# Evaluate the model on the test data
predictions = lr_model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="BETA", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Calculate the average beta from the predictions
average_beta = predictions.agg(avg("prediction")).collect()[0][0]
print(f"Average Beta (used as market return): {average_beta}")

# Add the average beta to the DataFrame
predictions = predictions.withColumn("market_return", col("prediction") * 0 + average_beta)

# Convert 'features' column to string (for CSV compatibility)
predictions = predictions.withColumn("features", predictions["features"].cast("string"))



lr_model.write().overwrite().save("hdfs://namenode:8020/model2")
# Stop Spark session
spark.stop()
