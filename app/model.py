from typing import List
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from.config import API_KEY, MODEL_NAME, TEMPERATURE
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd

def create_chain(dataframe):
    mistral = ChatGroq(temperature=TEMPERATURE, groq_api_key=API_KEY, model_name=MODEL_NAME)
    prompt_template = """
    You are an AI assistant skilled at analyzing tabular data and generating insights based on natural language queries.
    Given this tabular data: {data}
    And this query: {query}
    Please provide a detailed response that summarizes the key insights from the data relevant to the query. If necessary, you can perform calculations or aggregations on the data to derive insights. Your response should be tailored to the specific query and provide a thorough analysis of the data.
    """
    prompt = PromptTemplate(input_variables=["data", "query"], template=prompt_template)

    # Load the pre-trained embedding model
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create a list of Document objects from the dataframe and generate embeddings
    documents = [Document(page_content=str(row)) for row in dataframe.to_dict(orient="records")]
    embeddings = [embeddings_model.embed_query(doc.page_content) for doc in documents]

    # Create the FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Create the VectorStoreRetrieverMemory retriever
    retriever = vectorstore.as_retriever()
    memory_stream = []
    chain = ConversationalRetrievalChain.from_llm(
        llm=mistral,
        retriever=retriever,
        memory_stream=memory_stream,
        prompt=prompt
    )
    return chain
    prompt = create_prompt_template()

    return ConversationalRetrievalChain.from_llm(llm=mistral, retriever=retriever, memory_stream=[], prompt=prompt)
