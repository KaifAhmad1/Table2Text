# User Interface Script 
import streamlit as st
import pandas as pd
from chatbot import initialize_model, chatbot
from utils import display_data

def main():
    st.set_page_config(page_title="Tabular Data Chatbot", layout="wide")
    st.title("Chatbot with LLM Querying Tabular Data")
    
    initialize_model()

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        display_data(data)
        
        query = st.text_area("Enter your query", height=100)
        if st.button("Submit"):
            if query:
                with st.spinner("Processing..."):
                    result = chatbot(query, data)
                st.write("Result:")
                st.write(result)
            else:
                st.error("Please enter a query")
    else:
        st.info("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()

