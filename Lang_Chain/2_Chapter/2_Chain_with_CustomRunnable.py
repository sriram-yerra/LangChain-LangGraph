import os
from dotenv import load_dotenv

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


# ==============================
# Load environment variables
# ==============================
load_dotenv()

required_vars = [
    "GEMINI_MODEL",
    "TEMPERATURE",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_REGION",
    "MAX_OUTPUT_TOKENS"
]

for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"{var} not found in .env")

print("Vertex AI configuration loaded")


# ==============================
# TASK 1 — First Prompt
# ==============================
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}")
])


# ==============================
# TASK 2 — LLM
# ==============================
llm_vertexAI = ChatVertexAI(
    model_name=os.getenv("GEMINI_MODEL"),
    temperature=float(os.getenv("TEMPERATURE")),
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_REGION"),
    max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS")),
    convert_system_message_to_human=True
)


# ==============================
# TASK 3 — Output Parser
# ==============================
str_parser = StrOutputParser()  # LLMs return message objects.
'''
AIMessage(content="Paris")
Parser extracts only the text:
"Paris"
'''

# ==============================
# TASK 4 — Custom Runnable
# ==============================
def dictionary_maker(text: str) -> dict:
    return {"text": text}

dictionary_maker_runnable = RunnableLambda(dictionary_maker)


# ==============================
# TASK 5 — Second Prompt
# ==============================
prompt_post = ChatPromptTemplate.from_messages([
    ("system", "You're a social media post generator."),
    ("human", "Create a post for the following text for LinkedIn: {text}")
])


# ==============================
# CHAIN PIPELINE
# ==============================
chain = (
    prompt_template
    | llm_vertexAI
    | str_parser
    | dictionary_maker_runnable
    | prompt_post
    | llm_vertexAI
    | str_parser
)


result = chain.invoke({
    "input": "What is the capital of France?"
})

print(result)