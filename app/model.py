from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from .config import API_KEY, MODEL_NAME, TEMPERATURE
import faiss 
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

def create_chain(dataframe):
    mistral = ChatGroq(temperature=TEMPERATURE, groq_api_key=API_KEY, model_name=MODEL_NAME)
    prompt_template = """
    You are an AI assistant skilled at analyzing tabular data and generating insights based on natural language queries.
    Given this tabular data: {data}
    And this query: {query}
    Please provide a detailed response that summarizes the key insights from the data relevant to the query. If necessary, you can perform calculations or aggregations on the data to derive insights. Your response should be tailored to the specific query and provide a thorough analysis of the data.
    """
    prompt = PromptTemplate(input_variables=["data", "query"], template=prompt_template)

    # Create an InMemoryDocstore
    docstore = InMemoryDocstore(dataframe.to_dict(orient="records"))

    # Convert the docstore to a list of texts
    texts = [doc.page_content for doc in docstore]

    # Create the FAISS vector store
    vectorstore = FAISS.from_texts(texts)

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
