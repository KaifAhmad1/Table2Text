import pandas as pd
import streamlit as st

def dataframe_to_string(df):
    return df.to_string(index=False)

def display_data(df):
    st.dataframe(df, height=400)

def load_sample_data():
    return pd.read_csv("data/sample.csv")
