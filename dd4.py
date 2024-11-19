import os
import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressionModel, LinearRegressionModel
from pyspark.ml.feature import VectorAssembler
import plotly.express as px
import atexit

# Set environment variables for Spark and Java (ensure these paths are correct for your system)
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'

# Optional: Set Hadoop home for native libraries to avoid warnings (ensure you have Hadoop installed)
os.environ['HADOOP_HOME'] = '/opt/bitnami/hadoop'

# Initialize Spark Session
if 'spark' not in st.session_state:
    st.session_state.spark = SparkSession.builder \
        .appName("Stock Prediction and Visualization Dashboard") \
        .getOrCreate()

# Set Spark log level to avoid excessive log outputs in the console
spark = st.session_state.spark
spark.sparkContext.setLogLevel("WARN")  # Set to INFO for more detailed logs, or ERROR for fewer logs

# Load the trained Random Forest and Linear Regression models
rf_model_path = "hdfs://namenode:8020/model/random_forest_stock_model"
lr_model_path = "hdfs://namenode:8020/model2"

try:
    rf_model = RandomForestRegressionModel.load(rf_model_path)
except Exception as e:
    st.error(f"Failed to load Random Forest model: {e}")
    st.stop()

try:
    lr_model = LinearRegressionModel.load(lr_model_path)
except Exception as e:
    st.error(f"Failed to load Linear Regression model: {e}")
    st.stop()

# Load preprocessed data for visualization
data_path = "hdfs://namenode:8020/model/preprocessed_data"
try:
    df_spark = spark.read.csv(data_path, header=True, inferSchema=True)
    df = df_spark.toPandas()
except Exception as e:
    st.error(f"Failed to load preprocessed data: {e}")
    st.stop()

# Custom CSS for styling
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f4f7fb;
        }
        .sidebar .sidebar-content {
            background-color: #1d2d3d;
            color: white;
        }
        .stButton>button {
            background-color: #006d77;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #004f5c;
        }
        h1, h2, h3, h4 {
            font-family: 'Arial', sans-serif;
            color: #333333;
        }
        .stTextInput>div>input {
            border: 2px solid #006d77;
        }
        .stSelectbox, .stNumberInput, .stSlider {
            background-color: #e1f1f6;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation Menu
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Price Prediction", "Beta Prediction", "Visualization"])

if page == "Price Prediction":
    st.title("Stock Close Price Prediction")

    # Input fields for prediction
    col1, col2 = st.columns(2)
    with col1:
        open_value = st.number_input("Open", min_value=0.0, step=0.1)
        high_value = st.number_input("High", min_value=0.0, step=0.1)
        low_value = st.number_input("Low", min_value=0.0, step=0.1)
        volume = st.number_input("Volume", min_value=0.0, step=0.1)
        sma10 = st.number_input("SMA10", min_value=0.0, step=0.1)
        ema10 = st.number_input("EMA10", min_value=0.0, step=0.1)

    with col2:
        macd520 = st.number_input("MACD520", min_value=-1000.0, step=0.1)
        adx20 = st.number_input("ADX20", min_value=0.0, step=0.1)
        cci5 = st.number_input("CCI5", min_value=-1000.0, step=0.1)
        year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
        month = st.number_input("Month", min_value=1, max_value=12, step=1)
        day = st.number_input("Day", min_value=1, max_value=31, step=1)
        weekday = st.number_input("Weekday (1=Sun, 7=Sat)", min_value=1, max_value=7, step=1)

    if st.button("🔮 Predict Close Price"):
        user_input = pd.DataFrame([[open_value, high_value, low_value, volume, sma10, ema10,
                                    macd520, adx20, cci5, year, month, day, weekday]],
                                  columns=['open', 'high', 'low', 'volume', 'sma10', 'ema10',
                                           'macd520', 'ADX20', 'CCI5', 'year', 'month', 'day', 'weekday'])
        user_spark_df = spark.createDataFrame(user_input)
        assembler = VectorAssembler(
            inputCols=['open', 'high', 'low', 'volume', 'sma10', 'ema10',
                       'macd520', 'ADX20', 'CCI5', 'year', 'month', 'day', 'weekday'],
            outputCol="features")
        user_spark_df = assembler.transform(user_spark_df)
        prediction = rf_model.transform(user_spark_df)
        predicted_close = prediction.select("prediction").collect()[0][0]
        st.write(f"### Predicted Close Price: {predicted_close:.2f} USD")

elif page == "Beta Prediction":
    st.title("Beta Value Prediction")

    open_value = st.number_input("Open", min_value=0.0, step=1.0)
    high_value = st.number_input("High", min_value=0.0, step=1.0)
    low_value = st.number_input("Low", min_value=0.0, step=1.0)
    close_value = st.number_input("Close", min_value=0.0, step=1.0)
    volume = st.number_input("Volume", min_value=0.0, step=1.0)

    if st.button("Predict Beta Value"):
        user_input = pd.DataFrame([[open_value, high_value, low_value, close_value, volume]],
                                  columns=['open', 'high', 'low', 'close', 'volume'])
        user_spark_df = spark.createDataFrame(user_input)
        assembler = VectorAssembler(
            inputCols=['open', 'high', 'low', 'close', 'volume'],
            outputCol="features")
        user_spark_df = assembler.transform(user_spark_df)
        prediction = lr_model.transform(user_spark_df)
        predicted_beta = prediction.select("prediction").collect()[0][0]
        st.write(f"### Predicted Beta Value: {predicted_beta:.4f}")

elif page == "Visualization":
    st.title("Stock Data Visualization")
    st.sidebar.header("Filters")
    
    # Fixed years and month options
    selected_month = st.sidebar.multiselect("Select Month", options=list(range(1, 13)), default=list(range(1, 13)))
    
    # Fixed year logic based on selected months
    if all(month in range(1, 6) for month in selected_month):  # All months in range 1-5
        selected_year = [2015]
    elif all(month in range(6, 13) for month in selected_month):  # All months in range 6-12
        selected_year = [2014]
    else:
        selected_year = [2015, 2014]  # Mixed months selected
    
    # Filter the dataframe
    filtered_df = df[(df["year"].isin(selected_year)) & (df["month"].isin(selected_month))]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Line Chart: Closing Prices Over Time")
        line_chart = px.line(filtered_df, x="date", y="close", title="Closing Prices Over Time",
                             labels={"close": "Close Price", "date": "Date"})
        st.plotly_chart(line_chart)
    
    with col2:
        st.write("Bar Chart: Volume by Month")
        bar_chart = px.bar(filtered_df, x="month", y="volume", color="month", title="Volume by Month",
                           labels={"volume": "Volume", "month": "Month"})
        st.plotly_chart(bar_chart)
    
    with col1:
        st.write("Pie Chart: Average Indicators")
        pie_data = filtered_df[["sma10", "ema10", "macd520", "ADX20", "CCI5"]].mean().reset_index()
        pie_data.columns = ["Indicator", "Value"]
        pie_chart = px.pie(pie_data, names="Indicator", values="Value", title="Average Stock Indicators")
        st.plotly_chart(pie_chart)
    
    with col2:
        st.write("Scatter Plot: High vs Low Prices")
        scatter_plot = px.scatter(filtered_df, x="high", y="low", title="High vs Low Prices",
                                  labels={"high": "High Price", "low": "Low Price"})
        st.plotly_chart(scatter_plot)
