# App.py Module 
import pandas as pd
import streamlit as st
from app.model import create_chain

st.set_page_config(page_title="Conversational Data Analysis", layout="wide")

st.title("Conversational Data Analysis")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if 'memory_stream' not in st.session_state:
    st.session_state.memory_stream = []

if 'chain' not in st.session_state:
    st.session_state.chain = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Here is the data you uploaded:")
    st.write(df)

    if st.session_state.chain is None:
        st.session_state.chain = create_chain(df)

    query = st.text_input("Enter your query:")

    if query:
        result = st.session_state.chain.predict(query=query)
        st.session_state.memory_stream.append((query, result))

        st.subheader("Query History and Responses")
        for i, (q, r) in enumerate(st.session_state.memory_stream):
            st.markdown(f"**Query {i+1}:** {q}")
            st.markdown(f"**Response {i+1}:** {r}")
else:
    st.write("Please upload a CSV file to get started.")
