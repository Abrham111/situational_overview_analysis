from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def aggregate_per_customer(df):
  """
  Aggregates the following metrics per customer:
  - Average TCP retransmission (sum of DL and UL retransmission).
  - Average RTT (sum of DL and UL RTT).
  - Handset type (mode).
  - Average throughput (sum of DL and UL throughput).
  - User ID (Bearer Id).

  Parameters:
    df (pd.DataFrame): The input dataset.

  Returns:
    pd.DataFrame: Aggregated metrics per customer.
  """
  # Fill missing values with mean or mode
  df['TCP Retransmission'] = (
    df['TCP DL Retrans. Vol (Bytes)'] + df['TCP UL Retrans. Vol (Bytes)']
  ).fillna(0)
  df['RTT'] = (
    df['Avg RTT DL (ms)'] + df['Avg RTT UL (ms)']
  ).fillna(0)
  df['Throughput'] = (
    df['Avg Bearer TP DL (kbps)'] + df['Avg Bearer TP UL (kbps)']
  ).fillna(0)
  df['Handset Type'] = df['Handset Type'].fillna(df['Handset Type'].mode()[0])

  # Aggregate per customer
  aggregated = df.groupby('IMSI').agg({
    'TCP Retransmission': 'mean',
    'RTT': 'mean',
    'Throughput': 'mean',
    'Handset Type': lambda x: x.mode()[0],  # Most frequent handset type
    'Bearer Id': 'first'  # Assuming 'Bearer Id' is unique per customer
  }).reset_index()

  return aggregated

def compute_top_bottom_frequent(df, column, top_n=10):
  """
  Computes the top, bottom, and most frequent values of a column.

  Parameters:
      df (pd.DataFrame): The input dataset.
      column (str): The column name for which to compute values.
      top_n (int): The number of top and bottom values to compute.

  Returns:
      dict: A dictionary with keys 'top', 'bottom', and 'most_frequent'.
  """
  top = df[column].nlargest(top_n).tolist()
  bottom = df[column].nsmallest(top_n).tolist()
  most_frequent = df[column].value_counts().head(top_n).index.tolist()

  return {"top": top, "bottom": bottom, "most_frequent": most_frequent}


def compute_distribution(df, column, group_by):
  """
  Computes the distribution of a column grouped by another column.

  Parameters:
    df (pd.DataFrame): The input dataset.
    column (str): The column for which to compute the distribution (e.g., 'Throughput').
    group_by (str): The column to group by (e.g., 'Handset Type').

  Returns:
    pd.DataFrame: The mean values of the column grouped by the group_by column.
  """
  distribution = df.groupby(group_by)[column].mean().reset_index()
  return distribution

def perform_kmeans_clustering(df, columns, n_clusters=3):
  """
  Performs K-Means clustering based on selected columns.

  Parameters:
    df (pd.DataFrame): The input dataset.
    columns (list): List of column names to use for clustering.
    n_clusters (int): The number of clusters (k).

  Returns:
    pd.DataFrame: The original dataframe with an added 'Cluster' column.
    pd.DataFrame: Cluster centers for interpretation.
  """
  # Preprocessing: Handle missing values and standardize
  df = df.copy()
  for column in columns:
    df[column] = df[column].fillna(df[column].mean())
  
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(df[columns])
  
  # Perform k-means clustering
  kmeans = KMeans(n_clusters=n_clusters, random_state=42)
  df['Cluster'] = kmeans.fit_predict(scaled_data)
  
  cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                                 columns=columns)
  
  return df, cluster_centers
