import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from model import create_chain
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set the page configuration
st.set_page_config(page_title="Table2Text", page_icon=":bar_chart:", layout="wide")

# Add a dark theme
st.markdown("""
<style>
body {
    color: #fff;
    background-color: #111;
}
</style>
""", unsafe_allow_html=True)

# Add a logo or banner image
st.image("https://example.com/logo.png")

# Define the conversational chain
@st.cache_data
def conversational_chain(df, question):
    chain = create_chain(df)
    return chain.run(question=question)

# Add a title and description
st.title("Table2Text")
st.write("A Conversational Data Analysis Application specializing in analyzing your tabular data using natural language queries.")

# Handle CSV file upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Show a progress bar while the data is being loaded
    try:
        with st.spinner('Loading data...'):
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
    else:
        # Create tabs for different sections of the app
        tab1, tab2, tab3 = st.tabs(["Data Preview", "Query", "Data Exploration"])

        with tab1:
            st.subheader("Data Preview")
            st.write("Here you can preview and filter your data.")

            # Allow users to filter the data
            with st.expander("Filter Data"):
                column = st.selectbox("Select a column to filter", df.columns, key='filter_column')
                values = st.multiselect("Select values to include", df[column].unique(), key='filter_values')
                if values:
                    df_filtered = df[df[column].isin(values)]
                else:
                    df_filtered = df
                st.dataframe(df_filtered)

            # Allow users to download the filtered data
            csv = df_filtered.to_csv(index=False)
            st.download_button("Download filtered data", data=csv, file_name="filtered_data.csv", mime="text/csv")

        with tab2:
            st.subheader("Ask Your Query")
            st.write("Use natural language to ask questions about your data.")

            # Add a section for query input
            question = st.text_input("Enter your query", placeholder="Type your query here...", key='query_input')

            # Run the chain and display the response
            if question:
                with st.spinner('Processing your query...'):
                    try:
                        result = conversational_chain(df, question)
                        st.success(f"**Response:** {result}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        with tab3:
            st.subheader("Explore Your Data")
            st.write("Visualize your data with various plots and charts.")

            # Display statistical summaries
            st.subheader("Statistical Summary")
            st.write(df.describe())

            # Display scatter plot
            st.subheader("Scatter Plot")
            col1, col2 = st.columns(2)
            with col1:
                x_column_scatter = st.selectbox("Select a column for the x-axis", df.columns, key='scatter_x')
            with col2:
                y_column_scatter = st.selectbox("Select a column for the y-axis", df.columns, key='scatter_y')
            fig = px.scatter(df, x=x_column_scatter, y=y_column_scatter, hover_data=df.columns)
            st.plotly_chart(fig)

            # Display bar chart
            st.subheader("Bar Chart")
            col1, col2 = st.columns(2)
            with col1:
                x_column_bar = st.selectbox("Select a column for the x-axis", df.columns, key='bar_x')
            with col2:
                y_column_bar = st.selectbox("Select a column for the y-axis", df.columns, key='bar_y')
            fig = px.bar(df, x=x_column_bar, y=y_column_bar)
            st.plotly_chart(fig)

            # Display heatmap
            st.subheader("Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, ax=ax, cmap="coolwarm")
            st.pyplot(fig)
else:
    st.write("Please upload a CSV file to get started.")
