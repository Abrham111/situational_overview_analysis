import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
  return pd.read_csv(file_path)

def handle_missing_values(data):
  for col in data.select_dtypes(include=['float64', 'int64']).columns:
    data[col].fillna(data[col].mean(), inplace=True)
    return data

def describe_variables(data):
  description = data.describe()  # Summary statistics for numerical columns
  data_types = data.dtypes  # Data types for each column
  return description, data_types

def segment_by_deciles(data):
  data['Total Duration (s)'] = data['Dur. (ms)'] / 1000  # Convert duration to seconds
  total_duration_per_user = data.groupby('IMSI')['Total Duration (s)'].sum()
    
  # Decile segmentation based on total duration
  total_duration_per_user = total_duration_per_user.to_frame(name='Total Duration (s)')
  total_duration_per_user['Decile Class'] = pd.qcut(total_duration_per_user['Total Duration (s)'], 5, labels=False)
    
  # Compute total data (DL + UL)
  total_data_per_user = data.groupby('IMSI')['Total DL (Bytes)'].sum() + data.groupby('IMSI')['Total UL (Bytes)'].sum()
  total_duration_per_user['Total Data (DL+UL)'] = total_data_per_user
    
  return total_duration_per_user

def dispersion_parameters(data):
  # Handle missing values
  data = data.fillna(0)
  numerical_data = data.select_dtypes(include=['float64', 'int64'])
  dispersion = {
        'variance': numerical_data.var(),
        'std_deviation': numerical_data.std()
    }
  return pd.DataFrame(dispersion)

def graphical_analysis(data):
  num_cols = data.select_dtypes(include=['float64', 'int64']).columns
  for col in num_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

def bivariate_analysis(data):
  apps = ['Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
      'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Social Media DL (Bytes)', 'Social Media UL (Bytes)', 
      'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 'Email UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)']
  
  total_dl_ul = data['Total DL (Bytes)'] + data['Total UL (Bytes)']
  
  for app in apps:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[app], y=total_dl_ul)
    plt.title(f'Relationship Between {app} and Total DL+UL Data')
    plt.xlabel(app)
    plt.ylabel('Total DL + UL (Bytes)')
    plt.show()

  return

def correlation_analysis(data):
  apps = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 
      'Email DL (Bytes)', 'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 
      'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
      'Other DL (Bytes)', 'Other UL (Bytes)']
  
  app_data = data[apps]
  correlation_matrix = app_data.corr()
  
  plt.figure(figsize=(12, 8))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f')
  plt.title('Correlation Matrix for Application Data')
  plt.show()

def dimensionality_reduction(data):
  apps = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 
      'Email DL (Bytes)', 'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 
      'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
      'Other DL (Bytes)', 'Other UL (Bytes)']
  
  data_for_pca = data[apps].fillna(0)  # Fill missing values with 0 for PCA
  scaler = StandardScaler()
  data_scaled = scaler.fit_transform(data_for_pca)
  
  pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
  pca_result = pca.fit_transform(data_scaled)
  
  pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
  plt.figure(figsize=(10, 6))
  sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'])
  plt.title('PCA of Application Data')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.show()

  print("Explained variance ratio:", pca.explained_variance_ratio_)
  return pca_df

