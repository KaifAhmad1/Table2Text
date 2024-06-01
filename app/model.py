from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd

def create_chain(dataframe):
    # Initialize the language model
    mistral = ChatGroq(temperature=0.7, groq_api_key="YOUR_API_KEY", model_name="YOUR_MODEL_NAME")

    # Define the prompt template
    prompt_template = """
    You are an AI assistant skilled at analyzing tabular data and generating insights based on natural language queries.
    Given this tabular data: {data}
    And this query: {query}
    Please provide a detailed response that summarizes the key insights from the data relevant to the query. If necessary, you can perform calculations or aggregations on the data to derive insights. Your response should be tailored to the specific query and provide a thorough analysis of the data.
    """
    prompt = PromptTemplate(input_variables=["data", "query"], template=prompt_template)

    # Convert the dataframe to a list of Document objects
    documents = [Document(page_content=str(row)) for row in dataframe.to_dict(orient="records")]

    # Create the vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts([doc.page_content for doc in documents], embeddings, metadatas=None)
    retriever = vectorstore.as_retriever()

    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=mistral,
        retriever=retriever,
        prompt=prompt
    )

    return chain
