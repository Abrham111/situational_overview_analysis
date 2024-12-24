import streamlit as st
import sys
sys.path.append('../scripts')
from load_data import load_env_variables, create_db_engine, fetch_data

# Load data from a database
db_config = load_env_variables()
conn = create_db_engine(db_config)
query = "SELECT * FROM xdr_data;"
data = fetch_data(conn, query)

# Set the title of the Streamlit app
st.title("User Overview Dashboard")

# Display a table with some statistics for the selected user
st.write("User Statistics")
st.table(data.describe())
