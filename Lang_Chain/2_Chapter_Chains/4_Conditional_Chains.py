"""
Script Description:
Demonstrates Conditional Chains in LangChain using Vertex AI.

Pipeline:
1. Classifies a movie review as positive or negative.
2. Based on the classification, routes the input to different chains.
3. Generates either a LinkedIn post or Instagram caption.
"""

import os                                                     # Import OS module for environment variable access
import json                                                   # Import json for parsing model responses
from dotenv import load_dotenv                                # Import dotenv to load .env variables

from langchain_google_vertexai import ChatVertexAI            # Import Vertex AI Gemini chat model wrapper
from langchain_core.prompts import ChatPromptTemplate         # Import prompt template utility
from langchain_core.output_parsers import StrOutputParser     # Import parser to convert AIMessage → string
from langchain_core.runnables import RunnableLambda, RunnableBranch  # Import runnable utilities

from pydantic import BaseModel                                # Import Pydantic BaseModel for schema validation
from typing import Literal                                    # Import Literal type for fixed allowed values

# ========================================
# TASK 1 — Load Environment Variables
# ========================================
load_dotenv()                                                 # Load environment variables from .env file

required_vars = [                                             # Required Vertex AI configuration variables
    "GEMINI_MODEL",
    "TEMPERATURE",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_REGION",
    "MAX_OUTPUT_TOKENS"
]

for var in required_vars:                                     # Validate required variables
    if not os.getenv(var):                                    # Check if variable exists
        raise ValueError(f"{var} not found")                   # Raise error if missing

print("Vertex AI configuration loaded")                       # Confirm configuration

# ========================================
# TASK 2 — Initialize Vertex AI LLM
# ========================================
llm_vertexAI = ChatVertexAI(                                  # Initialize Gemini model
    model_name=os.getenv("GEMINI_MODEL"),                     # Model name
    temperature=float(os.getenv("TEMPERATURE")),              # Control randomness
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),                # Google Cloud project ID
    location=os.getenv("GOOGLE_CLOUD_REGION"),                # Vertex AI region
    max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS")),    # Token limit
    convert_system_message_to_human=True                      # Convert system messages for Gemini compatibility
)

# ========================================
# TASK 6 — Movie Review Classification Prompt
# ========================================
prompt_template = ChatPromptTemplate.from_messages([           # Prompt template for classification
    ("system", "You are a movie review evaluator."),
    ("human", """Please categorize the movie review as positive or negative.
      Return ONLY JSON in this format: {"movie_summary_flag": "positive"}
      Review: {input}"""
    )
])

str_parser = StrOutputParser()                                # Convert AIMessage → string

# ========================================
# TASK 3 — Define Structured Output Schema
# ========================================
class llm_schema(BaseModel):
    movie_summary_flag: Literal["positive", "negative"]       # Allowed classification values

# ========================================
# TASK 4 — Parse JSON → Pydantic Schema
# ========================================
def parse_llm_output(text: str) -> llm_schema:
    """
    Function Description:
    Parses JSON output from the LLM and converts it into
    the defined Pydantic schema.
    """
    data = json.loads(text)                                    # Convert JSON string → dictionary
    return llm_schema(**data)                                  # Convert dictionary → Pydantic model

json_parser = RunnableLambda(parse_llm_output)                 # Convert parsing function into Runnable

# ========================================
# TASK 5 — Extract Label From Pydantic
# ========================================
def pydantic_json(input: llm_schema) -> str:
    """
    Function Description:
    Extracts the movie_summary_flag value from the Pydantic object.
    """
    return input.model_dump()["movie_summary_flag"]            # Convert Pydantic model → dictionary

pydantic_json_lambda = RunnableLambda(pydantic_json)           # Convert function into Runnable


# ========================================
# TASK 5 — LinkedIn Chain
# ========================================
linkedin_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a LinkedIn post generator"),
    ("human", "Create a LinkedIn post about this movie review: {text}")
])

linkedin_chain = linkedin_prompt | llm_vertexAI | str_parser

# ========================================
# TASK 6 — Instagram Chain
# ========================================
instagram_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an Instagram caption generator"),
    ("human", "Create an Instagram caption about this movie review: {text}")
])

instagram_chain = instagram_prompt | llm_vertexAI | str_parser

# ========================================
# TASK 9 — Conditional Branching
# ========================================
def is_positive(text: str) -> bool:
    """
    Function Description:
    Checks whether the sentiment classification returned
    by the previous step is positive.
    """
    return "positive" in text

conditional_chain = RunnableBranch(                       # Conditional router
    (is_positive, linkedin_chain),                        # If positive → LinkedIn
    instagram_chain                                       # Otherwise → Instagram
)

# ========================================
# TASK 10 — Final Orchestrator Pipeline
# ========================================
final_orchestrator = (
    prompt_template                              # Step 1: Evaluate movie review
    | llm_vertexAI                               # Step 2: Generate JSON classification
    | str_parser                                 # Step 3: Extract text
    | json_parser                                # Step 4: Convert JSON → Pydantic schema
    | pydantic_json_lambda                       # Step 5: Extract label
    | conditional_chain                          # Step 6: Route to correct chain
)

# ========================================
# TASK 11 — Execute Pipeline
# ========================================
result = final_orchestrator.invoke({"input": "I loved this KGF movie"})  # Run pipeline

print(result)                                                             # Print generated output