"""
Script Description:
Demonstrates three ways to run LangChain chains using Vertex AI:
1. Manual invocation
2. LCEL chain using the pipe operator
3. RunnableSequence chain
"""

import os                                                     # Import OS module to access environment variables
from dotenv import load_dotenv                                # Import dotenv to load variables from .env file

# LangChain Vertex AI
from langchain_google_vertexai import ChatVertexAI            # Import Vertex AI chat model wrapper from LangChain

# LangChain components
from langchain_core.prompts import ChatPromptTemplate         # Import ChatPromptTemplate to create structured prompts
from langchain_core.output_parsers import StrOutputParser     # Import parser to convert AIMessage output to plain string
from langchain_core.runnables import RunnableSequence         # Import RunnableSequence for explicit chain construction


# ==============================
# Load environment variables
# ==============================
load_dotenv()                                                 # Load environment variables from the .env file

required_vars = [                                             # Define list of required environment variables
    "GEMINI_MODEL",
    "TEMPERATURE",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_REGION",
    "MAX_OUTPUT_TOKENS"
]

for var in required_vars:                                     # Iterate through required environment variables
    if not os.getenv(var):                                    # Check whether each variable exists
        raise ValueError(f"{var} not found in .env")           # Raise an error if any required variable is missing

print("Vertex AI configuration loaded")                       # Confirm configuration is successfully loaded

# ==============================
# TASK 1 — Prompt Template
# ==============================
prompt_template = ChatPromptTemplate.from_messages([          # Create a chat prompt template with system and human messages
    ("system", "You are a helpful assistant"),                 # System instruction defining assistant behavior
    ("human", "{input}")                                       # Placeholder for user input
])

# ==============================
# TASK 2 — Vertex AI LLM
# ==============================
llm_vertexAI = ChatVertexAI(                                   # Initialize Gemini model through Vertex AI
    model_name=os.getenv("GEMINI_MODEL"),                      # Load model name from environment variable
    temperature=float(os.getenv("TEMPERATURE")),               # Control randomness of model responses
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),                 # Specify Google Cloud project ID
    location=os.getenv("GOOGLE_CLOUD_REGION"),                 # Specify Vertex AI region
    max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS")),     # Set maximum token limit for responses
    convert_system_message_to_human=True                       # Convert system message to human for Gemini compatibility
)

# ==============================
# TASK 3 — Output Parser
# ==============================
str_parser = StrOutputParser()                                 # Initialize parser to convert AIMessage output to string

# ====================================================
# Chain Style-1: Manual Invocation
# ====================================================
template = prompt_template.invoke({"input": "What is the capital of France?"})   # Format prompt using the template
res = llm_vertexAI.invoke(template)                                              # Send formatted prompt to the LLM
final_result = res.content                                                       # Extract text content from AIMessage object
print("\nManual Invocation Result:")                                             # Print section header
print(final_result)                                                              # Print final LLM response

# ====================================================
# Chain Style-2: Chain Invocation using LCEL "|" Operator
# LCEL: Lang Chanin Expression Language
# ====================================================
chain1 = prompt_template | llm_vertexAI | str_parser                              # Create LCEL chain using pipe operator
result1 = chain1.invoke({"input": "What is the capital of France?"})              # Execute chain with input dictionary
print("\nLCEL Chain Result:")                                                     # Print section header
print(result1)                                                                    # Print parsed output string

# ====================================================
# Chain Style-3: RunnableSequence Chain
# ====================================================
chain2 = RunnableSequence(                                                        # Create chain using RunnableSequence
    prompt_template,                                                              # First step: prompt template
    llm_vertexAI,                                                                 # Second step: LLM execution
    str_parser                                                                    # Third step: parse output to string
)

result2 = chain2.invoke({"input": "What is the capital of France?"})              # Execute RunnableSequence chain
print("\nRunnableSequence Result:")                                               # Print section header
print(result2)                                                                    # Print parsed response

