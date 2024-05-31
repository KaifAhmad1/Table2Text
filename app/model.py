from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from .config import API_KEY, MODEL_NAME, TEMPERATURE

def create_chain(dataframe):
    mistral = ChatGroq(temperature=TEMPERATURE, groq_api_key=API_KEY, model_name=MODEL_NAME)

    prompt_template = """
    You are an AI assistant skilled at analyzing tabular data and generating insights based on natural language queries.

    Given this tabular data: {data}

    And this query: {query}

    Please provide a detailed response that summarizes the key insights from the data relevant to the query. If necessary, you can perform calculations or aggregations on the data to derive insights. Your response should be tailored to the specific query and provide a thorough analysis of the data.
    """
    prompt = PromptTemplate(input_variables=["data", "query"], template=prompt_template)

    memory_stream = []
    chain = ConversationalRetrievalChain.from_llm(
        llm=mistral,
        retriever=dataframe.to_dict(orient="records"),
        memory_stream=memory_stream,
        prompt=prompt
    )
    return chain
