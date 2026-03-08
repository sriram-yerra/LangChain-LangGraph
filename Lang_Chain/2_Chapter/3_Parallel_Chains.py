import os
from dotenv import load_dotenv

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

load_dotenv()

# Vertex config check
required_vars = [
    "GEMINI_MODEL",
    "TEMPERATURE",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_REGION",
    "MAX_OUTPUT_TOKENS"
]

for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"{var} not found")

print("Vertex AI configuration loaded")

# LLM
llm_vertexAI = ChatVertexAI(
    model_name=os.getenv("GEMINI_MODEL"),
    temperature=float(os.getenv("TEMPERATURE")),
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_REGION"),
    max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS")),
    convert_system_message_to_human=True
)

# Movie summary prompt
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a movie summarizer"),
    ("human", "Please summarize the movie in brief : {input}")
])

str_parser = StrOutputParser()

# Convert text → dict
def dictionary_maker(text: str) -> dict:
    return {"text": text}

dictionary_maker_runnable = RunnableLambda(dictionary_maker)

# --------------------------------------------------------

# LinkedIn prompt
linkedin_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a LinkedIn post generator"),
    ("human", "Create a post for the following text for LinkedIn: {text}")
])

chain_linkedin = linkedin_prompt | llm_vertexAI | str_parser

# Instagram chain
def insta_chain(text: dict):
    text = text["text"]

    insta_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an Instagram caption generator"),
        ("human", "Create a post for the following text for Instagram: {text}")
    ])

    chain = insta_prompt | llm_vertexAI | str_parser
    return chain.invoke({"text": text})

insta_chain_runnable = RunnableLambda(insta_chain)

# Final parallel chain
final_chain = (
    prompt_template
    | llm_vertexAI
    | str_parser
    | dictionary_maker_runnable
    | RunnableParallel(
        branches={
            "linkedin": chain_linkedin,
            "instagram": insta_chain_runnable
        }
    )
)

result = final_chain.invoke({"input": "KGF"})
print(result)