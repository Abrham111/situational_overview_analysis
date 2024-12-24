import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mysql.connector
from dotenv import load_dotenv
import os

less_engaged_center = np.array([1.043550, 107473.680857, 4.306959e+07, 4.534280e+08, 4.964976e+08])  # Sessions Frequency	Total Duration (ms)	Total UL (Bytes)	Total DL (Bytes)	Total Traffic (Bytes)
worst_experience_center = np.array([42213190, 107.095, 62901.673])  # Cluster 1 from experience_analytics

# Calculate Scores

def calculate_engagement_score(data):
  """
  Calculate the engagement score based on Euclidean distance from the reference point (less_engaged_center).
  """
  data['engagement_score'] = data.apply(
    lambda row: np.linalg.norm(row[['Sessions Frequency', 'Total Duration (ms)', 'Total UL (Bytes)', 'Total DL (Bytes)', 'Total Traffic (Bytes)']] - less_engaged_center), axis=1
  )
  return data

def calculate_experience_score(data):
  """
  Calculate the experience score based on Euclidean distance from the reference point (worst_experience_center).
  """
  data['experience_score'] = data.apply(
    lambda row: np.linalg.norm(row[['TCP Retransmission', 'RTT', 'Throughput']] - worst_experience_center), axis=1
  )
  return data

def calculate_satisfaction(data):
  data['satisfaction_score'] = (data['engagement_score'] + data['experience_score']) / 2
  top_10_customers = data.nlargest(10, 'satisfaction_score')
  print("Top 10 Satisfied Customers:")
  print(top_10_customers[['Bearer Id', 'satisfaction_score']])
  return data, top_10_customers

def regression_model(data):
  features = data[['Total Duration (ms)', 'Total Traffic (Bytes)']]  # relevant features
  satisfaction = data['satisfaction_score']

  model = LinearRegression()
  model.fit(features, satisfaction)

  predictions = model.predict(features)
  # Calculate RMSE
  rmse = mean_squared_error(data['satisfaction_score'], predictions, squared=False)
  print("Regression Model RMSE:", rmse)
  # Compare predictions to actual scores
  comparison = pd.DataFrame({
    'Actual': data['satisfaction_score'],
    'Predicted': predictions
  })

  print(comparison)
  return model


# K means clustering
def perform_kmeans(data):
  kmeans = KMeans(n_clusters=2, random_state=42)
  kmeans.fit(data[['engagement_score', 'experience_score']])
  data['cluster'] = kmeans.labels_
  return data

# Aggregate the average satisfaction & experience score per cluster
def aggregate_scores(data):
  cluster_agg = data.groupby('cluster').agg(
    avg_satisfaction_score=('satisfaction_score', 'mean'),
    avg_experience_score=('experience_score', 'mean')
  ).reset_index()
  print("Cluster Aggregations:")
  print(cluster_agg)
  return cluster_agg

# Connect to the database and export the data
# Get the MySQL credentials from the .env file
host = os.getenv("MYSQL_HOST")
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
database = os.getenv("MYSQL_DATABASE")

def export_to_mysql(data):
  data = data.where(pd.notnull(data), None)  # replace NaN with None for MySQL
  
  # Rename columns to be database-friendly
  data = data.rename(columns={
    'Bearer Id': 'Bearer_Id',
    'satisfaction_score': 'satisfaction_score',
    'engagement_score': 'engagement_score',
    'experience_score': 'experience_score'
  })

  conn = mysql.connector.connect(
    host=host,
    user=user,
    password=password,
    database=database
  )
  cursor = conn.cursor()

  # Create table (if not exists)
  cursor.execute("""
  CREATE TABLE IF NOT EXISTS user_scores (
    Bearer_Id FLOAT,
    engagement_score FLOAT,
    experience_score FLOAT,
    satisfaction_score FLOAT
  )
  """)

  # Insert data into MySQL table
  for index, row in data.iterrows():
    cursor.execute("""
    INSERT INTO user_scores (Bearer_Id, engagement_score, experience_score, satisfaction_score)
    VALUES (%s, %s, %s, %s)
    """, (row['Bearer_Id'], row['engagement_score'], row['experience_score'], row['satisfaction_score']))

  conn.commit()

  # Query the table and display results
  cursor.execute("SELECT * FROM user_scores LIMIT 10;")
  result = cursor.fetchall()
  print("Exported Data (First 10 Rows):")
  for row in result:
    print(row)

  cursor.close()
  conn.close()
