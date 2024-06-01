import pandas as pd
import streamlit as st
from app.model import create_chain

# Initialize the Streamlit app with title and description
st.set_page_config(page_title="Conversational Data Analysis", page_icon=":bar_chart:", layout="wide")
st.title("Conversational Data Analysis")
st.write("Analyze your tabular data using natural language queries.")

# Add a logo or banner image
st.image("https://example.com/logo.png")

# Define the conversational chain
@st.cache
def conversational_chain(df, query):
    chain = create_chain(df)
    return chain.run(query=query, data=df)

# Handle CSV file upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Show a progress bar while the data is being loaded
    with st.spinner('Loading data...'):
        df = pd.read_csv(uploaded_file)

    # Create tabs for different sections of the app
    tab1, tab2 = st.tabs(["Data Preview", "Query"])

    with tab1:
        # Allow users to filter the data
        column = st.selectbox("Select a column to filter", df.columns)
        values = st.multiselect("Select values to include", df[column].unique())
        df = df[df[column].isin(values)]

        # Display a data preview with interactive options
        st.dataframe(df)

        # Display an interactive chart
        column = st.selectbox("Select a column to display", df.columns)
        st.line_chart(df[column])

        # Allow users to download the data
        csv = df.to_csv(index=False)
        st.download_button("Download data", data=csv, file_name="data.csv", mime="text/csv")

    with tab2:
        # Add a section for query input
        st.header("Ask your query")
        query = st.text_input("Enter your query", placeholder="Type your query here...")

        # Run the chain and display the response
        if query:
            try:
                result = conversational_chain(df, query)
                st.write(f"**Response:** {result}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file to get started.")
    st.balloons()
