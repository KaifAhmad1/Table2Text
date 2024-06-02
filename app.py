import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from app.model import create_chain
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set the page configuration
def set_page_config():
    st.set_page_config(page_title="Table2Text", page_icon=":bar_chart:", layout="wide")
    st.markdown("""
    <style>
    body {
        color: #fff;
        background-color: #111;
    }
    </style>
    """, unsafe_allow_html=True)
    st.image("https://example.com/logo.png")

# Define the conversational chain
@st.cache_data
def conversational_chain(df, question, chat_history=[]):
    chain = create_chain(df)
    return chain.run(question=question, chat_history=chat_history)

# Handle CSV file upload
def handle_file_upload():
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        try:
            with st.spinner('Loading data...'):
                df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}")
        else:
            return df

# Display data preview
def display_data_preview(df):
    st.subheader("Data Preview")
    st.write("Here you can preview and filter your data.")

    with st.expander("Filter Data"):
        column = st.selectbox("Select a column to filter", df.columns, key='filter_column')
        values = st.multiselect("Select values to include", df[column].unique(), key='filter_values')
        if values:
            df_filtered = df[df[column].isin(values)]
        else:
            df_filtered = df
        st.dataframe(df_filtered)

    csv = df_filtered.to_csv(index=False)
    st.download_button("Download filtered data", data=csv, file_name="filtered_data.csv", mime="text/csv")

# Display query section
def display_query_section(df):
    st.subheader("Ask Your Query")
    st.write("Use natural language to ask questions about your data.")

    for message in st.session_state.get('chat_history', []):
        st.write(message)

    question = st.text_input("Enter your query", placeholder="Type your query here...", key='query_input')

    if question:
        with st.spinner('Processing your query...'):
            try:
                result = conversational_chain(df, question, chat_history=st.session_state.get('chat_history', []))
                st.success(f"**Response:** {result}")

                st.session_state['chat_history'] = st.session_state.get('chat_history', []) + [f"**Query:** {question}", f"**Response:** {result}"]
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if 'chat_history' in st.session_state:
        follow_up_question = st.text_input("Ask a follow-up question", placeholder="Type your follow-up question here...", key='follow_up_input')

        if follow_up_question:
            with st.spinner('Processing your follow-up question...'):
                try:
                    follow_up_result = conversational_chain(df, follow_up_question, chat_history=st.session_state.get('chat_history', []))
                    st.success(f"**Follow-up Response:** {follow_up_result}")

                    st.session_state['chat_history'] = st.session_state.get('chat_history', []) + [f"**Follow-up Query:** {follow_up_question}", f"**Follow-up Response:** {follow_up_result}"]
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Display data exploration section
def display_data_exploration(df):
    st.subheader("Explore Your Data")
    st.write("Visualize your data with various plots and charts.")

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Scatter Plot")
    col1, col2 = st.columns(2)
    with col1:
        x_column_scatter = st.selectbox("Select a column for the x-axis", df.columns, key='scatter_x')
    with col2:
        y_column_scatter = st.selectbox("Select a column for the y-axis", df.columns, key='scatter_y')
    fig = px.scatter(df, x=x_column_scatter, y=y_column_scatter, hover_data=df.columns)
    st.plotly_chart(fig)

    st.subheader("Bar Chart")
    col1, col2 = st.columns(2)
    with col1:
        x_column_bar = st.selectbox("Select a column for the x-axis", df.columns, key='bar_x')
    with col2:
        y_column_bar = st.selectbox("Select a column for the y-axis", df.columns, key='bar_y')
    fig = px.bar(df, x=x_column_bar, y=y_column_bar)
    st.plotly_chart(fig)

    st.subheader("Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, ax=ax, cmap="coolwarm")
    st.pyplot(fig)

# Main function
def main():
    set_page_config()

    st.title("Table2Text")
    st.write("A Conversational Data Analysis Application specializing in analyzing your tabular data using natural language queries.")

    df = handle_file_upload()

    if df is not None:
        tab1, tab2, tab3 = st.tabs(["Data Preview", "Query", "Data Exploration"])

        with tab1:
            display_data_preview(df)

        with tab2:
            display_query_section(df)

        with tab3:
            display_data_exploration(df)
    else:
        st.write("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()
