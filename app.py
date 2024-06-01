# app.py
import pandas as pd
from chainlit import Chainlit, Image
from app.model import create_chain

chainlit = Chainlit(title="Conversational Data Analysis", description="Analyze your tabular data using natural language queries.")

# Add a logo or banner image
chainlit.preview_markdown("![Logo](https://example.com/logo.png)")

@chainlit.add_chain
def conversational_chain(df, query):
    chain = create_chain(df)
    return chain.run(query=query, data=df)

# Upload CSV file
uploaded_file = chainlit.upload_file("Upload your CSV file", file_types=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display a data preview with interactive options
    chainlit.data_preview(df, ignore_columns=["id"], display_rows=5, display_columns=4, column_filters=True, column_sorting=True, column_selection=True)

    # Add a section for query input
    chainlit.markdown("## Ask your query")
    query = chainlit.input_text("Enter your query", placeholder="Type your query here...")

    # Run the chain and display the response
    if query:
        result = conversational_chain(df, query)
        chainlit.markdown(f"**Response:** {result}")

    # Display conversation history
    chainlit.markdown("## Conversation History")
    with chainlit.expander("View Conversation History"):
        for i, (q, r) in enumerate(chainlit.session.memory.values(), start=1):
            chainlit.markdown(f"**Query {i}:** {q}")
            chainlit.markdown(f"**Response {i}:** {r}")
else:
    chainlit.preview_text("Please upload a CSV file to get started.")

chainlit.run()
