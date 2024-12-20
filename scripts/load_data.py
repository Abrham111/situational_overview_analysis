from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine

def load_env_variables():
  load_dotenv()
  db_config = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
  }
  return db_config

def create_db_engine(db_config):
  try:
    engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}")
    print("Database engine created successfully!")
    return engine
  except Exception as e:
    print(f"Error creating database engine: {e}")
    return None

def fetch_data(engine, query):
  try:
    df = pd.read_sql_query(query, engine)
    print("Data loaded successfully!")
    return df
  except Exception as e:
    print(f"Error loading data: {e}")
    return None
