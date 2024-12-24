import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def aggregate_metrics(df):
  """Aggregate metrics per customer ID."""
  metrics = df.groupby('MSISDN/Number').agg({
    'Bearer Id': 'count',  # Sessions frequency
    'Dur. (ms)': 'sum',   # Total session duration
    'Total UL (Bytes)': 'sum',
    'Total DL (Bytes)': 'sum'  # Total traffic (upload + download)
  }).reset_index()

  # Create a copy of the 'Bearer Id' column with the raw values
  bearer_ids = df.groupby('MSISDN/Number')['Bearer Id'].first().reset_index(name='Bearer Id')

  metrics.rename(columns={
    'Bearer Id': 'Sessions Frequency',
    'Dur. (ms)': 'Total Duration (ms)',
    'Total UL (Bytes)': 'Total UL (Bytes)',
    'Total DL (Bytes)': 'Total DL (Bytes)'
  }, inplace=True)
  metrics['Total Traffic (Bytes)'] = metrics['Total UL (Bytes)'] + metrics['Total DL (Bytes)']
  # Merge the metrics DataFrame with the Bearer Id List
  metrics = pd.merge(metrics, bearer_ids, on='MSISDN/Number', how='left')
  return metrics

def top_customers(metrics, metric):
  """Get top 10 customers by a specific metric."""
  return metrics.nlargest(10, metric)

def normalize_metrics(metrics):
  """Normalize metrics for clustering."""
  scaler = MinMaxScaler()
  norm_metrics = scaler.fit_transform(metrics[['Sessions Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']])
  return pd.DataFrame(norm_metrics, columns=['Sessions Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)'], index=metrics.index)

def perform_kmeans(norm_metrics, n_clusters=3):
  """Perform K-Means clustering."""
  kmeans = KMeans(n_clusters=n_clusters, random_state=42)
  labels = kmeans.fit_predict(norm_metrics)
  return labels, kmeans

def cluster_statistics(metrics):
  """Compute statistics for each cluster."""
  return metrics.groupby('Cluster').agg(['min', 'max', 'mean', 'sum'])

def plot_clusters(norm_metrics, labels):
  plt.figure(figsize=(10, 6))
  plt.scatter(norm_metrics.iloc[:, 0], norm_metrics.iloc[:, 1], c=labels, cmap='viridis', s=50)
  plt.title('Customer Clusters')
  plt.xlabel('Normalized Sessions Frequency')
  plt.ylabel('Normalized Total Duration (ms)')
  plt.colorbar(label='Cluster')
  plt.show()

def aggregate_application_traffic(df):
  """Aggregate user traffic per application."""
  app_columns = [col for col in df.columns if 'DL (Bytes)' in col or 'UL (Bytes)' in col]
  return df.groupby('MSISDN/Number')[app_columns].sum()

def top_users_per_application(application_traffic, application):
  """Get top 10 users by a specific application."""
  return application_traffic.nlargest(10, application)

def plot_top_applications(application_totals):
  """Plot the top 8 most used applications."""
  plt.figure(figsize=(10, 6))
  application_totals.head(8).plot(kind='bar', color=['skyblue', 'orange', 'green'])
  plt.title('Top 3 Most Used Applications')
  plt.ylabel('Total Traffic (Bytes)')
  plt.xlabel('Applications')
  plt.show()

def elbow_method(norm_metrics):
  """Use the Elbow Method to find the optimal number of clusters."""
  inertia = []
  for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(norm_metrics)
    inertia.append(kmeans.inertia_)
  plt.figure(figsize=(10, 6))
  plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
  plt.title('Elbow Method')
  plt.xlabel('Number of Clusters')
  plt.ylabel('Inertia')
  plt.show()

