from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.memory import ConversationBufferMemory
from langchain.schema.document import Document
import os
from app.config import API_KEY, MODEL_NAME, TEMPERATURE


def create_chain(data):
    # Initialize the language model
    mistral = ChatGroq(temperature=TEMPERATURE, groq_api_key=API_KEY, model_name=MODEL_NAME)

    # Define the prompt template
    prompt_template = """
    You are an AI assistant skilled at analyzing tabular data and generating insights based on natural language queries.
    Given this query: {question}
    And the following data: {context}
    Please provide a detailed response that summarizes the key insights from the data relevant to the query. If necessary, you can perform calculations or aggregations on the data to derive insights. Your response should be tailored to the specific query and provide a thorough analysis of the data.
    Here are some examples of good queries:
    - What is the average age of employees in each department?
    - What is the correlation between salary and years of experience?
    - Which department has the highest turnover rate?
    Here are some examples of bad queries:
    - Can you tell me a joke?
    - What is the weather like today?
    - Who won the last World Cup?
    Here are some examples of how to interpret the data:
    - To calculate the average age of employees in each department, group the data by department and calculate the mean age for each group.
    - To calculate the correlation between salary and years of experience, use a statistical method such as Pearson correlation coefficient.
    - To identify the department with the highest turnover rate, calculate the turnover rate for each department and compare the rates.
    Here are some specific instructions for numerical calculations:
    - Round numbers to two decimal places when presenting results.
    - Use appropriate statistical methods for data aggregation and analysis.
    - Handle missing values, outliers, and invalid inputs appropriately.
    Chat History:
    {chat_history}
    """
    prompt = PromptTemplate(input_variables=["question", "chat_history", "context"], template=prompt_template)

    # Convert the dataframe to a list of Document objects
    documents = [Document(page_content=str(row), metadata={'source': 'TableData.csv'}) for row in data.to_dict(orient="records")]

    # Create the vector store and retriever
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = BM25Retriever.from_documents(documents)

    # Create the memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=mistral,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain
