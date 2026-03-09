"""
Script Description:
Demonstrates Conditional Chains in LangChain.
The pipeline:
1. Classifies a movie review as positive or negative.
2. Based on the classification, routes the input to different chains.
3. Generates either a LinkedIn post or Instagram caption.
"""

import os                                                     # Import OS module for environment variable access
from dotenv import load_dotenv                                # Import dotenv to load .env variables

from langchain_openai import ChatOpenAI                       # Import OpenAI chat model wrapper
from langchain_core.prompts import ChatPromptTemplate         # Import prompt template utility
from langchain_core.output_parsers import StrOutputParser     # Import parser to convert AIMessage → string
from langchain_core.runnables import RunnableLambda, RunnableBranch  # Import runnable utilities

from pydantic import BaseModel                                # Import Pydantic BaseModel for schema validation
from typing import Literal                                    # Import Literal type for fixed allowed values

# ========================================
# TASK 1 — Load Environment Variables
# ========================================
load_dotenv()                                                 # Load environment variables from .env file

required_vars = [                                             # List of required environment variables for Vertex AI
    "GEMINI_MODEL",
    "TEMPERATURE",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_REGION",
    "MAX_OUTPUT_TOKENS"
]

for var in required_vars:                                     # Iterate through each required variable
    if not os.getenv(var):                                    # Check whether the variable exists
        raise ValueError(f"{var} not found")                   # Raise error if variable is missing

print("Vertex AI configuration loaded")                       # Confirm configuration loaded successfully

# ========================================
# TASK 2 — Initialize LLM
# ========================================
llm_vertexAI = ChatVertexAI(                                  # Initialize Gemini model through Vertex AI
    model_name=os.getenv("GEMINI_MODEL"),                     # Load model name from environment variables
    temperature=float(os.getenv("TEMPERATURE")),              # Control randomness of generated responses
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),                # Specify Google Cloud project ID
    location=os.getenv("GOOGLE_CLOUD_REGION"),                # Specify Vertex AI region
    max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS")),    # Set maximum number of tokens for model output
    convert_system_message_to_human=True                      # Convert system messages to human messages for Gemini compatibility
)

# ========================================
# TASK 3 — Define Structured Output Schema
# ========================================
class llm_schema(BaseModel):
    movie_summary_flag: Literal["positive", "negative"]                # Force LLM output to be either positive or negative

llm_structured_output = llm_openai.with_structured_output(llm_schema)  # Wrap LLM to return Pydantic structured output

# ========================================
# TASK 4 — Movie Review Classification Prompt
# ========================================
prompt_template = ChatPromptTemplate.from_messages([          # Create prompt template for review classification
    ("system", "You are a movie review evaluator"),           # Define system role
    ("human", "Please categorize the movie review as positive or negative : {input}")  # Insert user review
])

# ========================================
# TASK 5 — Convert Pydantic Output → String
# ========================================
def pydantic_json(input: llm_schema) -> str:
    """
    Function Description:
    Extracts the 'movie_summary_flag' value from the Pydantic object.
    """
    return input.model_dump()['movie_summary_flag']           # Convert Pydantic object to dict and return flag value

pydantic_json_lambda = RunnableLambda(pydantic_json)          # Convert Python function into Runnable

# ========================================
# TASK 6 — LinkedIn Post Chain
# ========================================
linkedin_prompt = ChatPromptTemplate.from_messages([                        # Prompt template for LinkedIn post generation
    ("system", "You are a LinkedIn post generator"),                        # Define assistant role
    ("human", "Create a post for the following text for LinkedIn: {text}")  # Insert content
])

str_parser = StrOutputParser()                                # Convert AIMessage output to string

chain_linkedin = linkedin_prompt | llm_openai | str_parser    # Create LinkedIn chain: Prompt → LLM → Parser

# ========================================
# TASK 7 — Instagram Chain
# ========================================
def insta_chain(data: dict):
    """
    Function Description:
    Generates an Instagram caption from the input text.
    """
    text = data["text"]                                       # Extract text value from dictionary input
    insta_prompt = ChatPromptTemplate.from_messages([          # Create prompt template for Instagram caption
        ("system", "You are a LinkedIn post generator"),      # Define assistant role
        ("human", "Create a post for the following text for Instagram: {text}")  # Insert text
    ])
    chain_insta = insta_prompt | llm_openai | str_parser       # Create Instagram generation chain
    result = chain_insta.invoke(text)                          # Execute Instagram chain
    return result                                              # Return generated caption

insta_chain_runnable = RunnableLambda(insta_chain)             # Convert Instagram function into Runnable

# ========================================
# TASK 8 — Conditional Branching
# ========================================
conditional_chain = RunnableBranch(                            # Create conditional router
    (lambda x: "positive" in x, chain_linkedin),               # If review is positive → run LinkedIn chain
    insta_chain_runnable                                       # Otherwise → run Instagram chain
)

# ========================================
# TASK 9 — Final Orchestrator Pipeline
# ========================================
final_orchestrator = (                                         # Build full pipeline
    prompt_template                                            # Step 1: Evaluate movie review
    | llm_structured_output                                    # Step 2: Generate structured classification
    | pydantic_json_lambda                                     # Step 3: Extract classification label
    | conditional_chain                                        # Step 4: Route to correct content generator
)

# ========================================
# TASK 10 — Execute Pipeline
# ========================================
result = final_orchestrator.invoke({"input": "I loved this KGF movie"})  # Run pipeline
print(result)                                                            # Print generated output