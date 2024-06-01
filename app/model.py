from typing import List
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from.config import API_KEY, MODEL_NAME, TEMPERATURE
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import pandas as pd

def convert_dataframe_to_documents(dataframe: pd.DataFrame) -> List[Document]:
    """Convert a DataFrame into a list of Document objects."""
    return [Document(page_content=str(row)) for row in dataframe.to_dict(orient="records")]

def initialize_chat_groq() -> ChatGroq:
    """Initialize the ChatGroq instance."""
    return ChatGroq(temperature=TEMPERATURE, groq_api_key=API_KEY, model_name=MODEL_NAME)

def create_prompt_template() -> PromptTemplate:
    """Create the PromptTemplate object."""
    template_text = """
    You are an AI assistant skilled at analyzing tabular data and generating insights based on natural language queries.
    Given this tabular data: {data}
    And this query: {query}
    Please provide a detailed response that summarizes the key insights from the data relevant to the query. If necessary, you can perform calculations or aggregations on the data to derive insights. Your response should be tailored to the specific query and provide a thorough analysis of the data.
    """
    return PromptTemplate(input_variables=["data", "query"], template=template_text)

def create_conversational_retrieval_chain(dataframe: pd.DataFrame) -> ConversationalRetrievalChain:
    """Set up a conversational retrieval chain using LangChain."""
    documents = convert_dataframe_to_documents(dataframe)
    vectorstore = FAISS.from_documents(documents)
    retriever = vectorstore.as_retriever()
    mistral = initialize_chat_groq()
    prompt = create_prompt_template()

    return ConversationalRetrievalChain.from_llm(llm=mistral, retriever=retriever, memory_stream=[], prompt=prompt)
