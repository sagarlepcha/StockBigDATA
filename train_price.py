import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofmonth, dayofweek, hour, minute
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import os

# Set environment variables for Spark and Java
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("OptimizedStockPrediction").config("spark.executor.memory", "8g").config("spark.executor.cores", "4").config("spark.network.timeout", "600s").config("spark.sql.shuffle.partitions", "200").getOrCreate()

# Load the dataset
data_path = "hdfs://namenode:8020/input/TCS_with_indicators_.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Convert 'date' column to timestamp and extract features
df = df.withColumn("date", col("date").cast("timestamp"))
df = df.withColumn("year", year(col("date"))) \
       .withColumn("month", month(col("date"))) \
       .withColumn("day", dayofmonth(col("date"))) \
       .withColumn("weekday", dayofweek(col("date"))) \
       .withColumn("hour", hour(col("date"))) \
       .withColumn("minute", minute(col("date")))

# Convert to Pandas DataFrame for preprocessing
data = df.toPandas()

# 1. Print data types of the columns
print("Data Types of Columns:")
print(data.dtypes)

# 2. Find and print the sum of null values in each column
print("\nSum of Null Values in Each Column:")
print(data.isnull().sum())

# 3. Remove rows containing null values
data_cleaned = data.dropna()

# Check after dropping null values
print("\nData after removing rows with null values:")
print(data_cleaned.isnull().sum())

# 4. Print sum of duplicate values and remove them
duplicate_count = data_cleaned.duplicated().sum()
print(f"\nSum of Duplicate Rows: {duplicate_count}")
data_cleaned = data_cleaned.drop_duplicates()

# Check after removing duplicates
duplicate_count_after = data_cleaned.duplicated().sum()
print(f"\nSum of Duplicate Rows after removal: {duplicate_count_after}")

# 5. Feature Engineering - Select relevant features
features = ['open', 'high', 'low', 'volume', 'sma10', 'ema10', 'macd520', 'ADX20', 'CCI5', 'year', 'month', 'day', 'weekday']
assembler = VectorAssembler(inputCols=features, outputCol="features")
df_spark = spark.createDataFrame(data_cleaned)  # Create Spark DataFrame from cleaned data
df_spark = assembler.transform(df_spark)

# 6. Split into train and test sets
train_data = df_spark.sample(fraction=0.8, seed=42)
test_data = df_spark.subtract(train_data)

# 7. Build the Random Forest Model
rf = RandomForestRegressor(featuresCol="features", labelCol="close", numTrees=100)

# 8. Train the model
rf_model = rf.fit(train_data)

# 9. Make predictions
predictions = rf_model.transform(test_data)

# 10. Evaluate the model
evaluator_rmse = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

# Print the evaluation metrics
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# 12. Save the Random Forest model directly to HDFS
hdfs_path = "hdfs://namenode:8020/model/random_forest_stock_model"
rf_model.write().overwrite().save(hdfs_path)

# Save Preprocessed Data to HDFS
preprocessed_spark_df = spark.createDataFrame(data_cleaned)
preprocessed_data_hdfs_path = "hdfs://namenode:8020/model/preprocessed_data"
preprocessed_spark_df.write.mode("overwrite").csv(preprocessed_data_hdfs_path, header=True)

print(f"Preprocessed data saved to HDFS at {preprocessed_data_hdfs_path}")

spark.stop()
