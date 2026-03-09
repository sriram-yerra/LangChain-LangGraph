"""
Script Description:
Demonstrates a multi-step LangChain pipeline using Vertex AI where:
1. A question is answered by the LLM.
2. The answer is converted into a dictionary using a custom runnable.
3. A second prompt generates a LinkedIn post based on the answer.
"""

import os                                                     # Import OS module to access environment variables
from dotenv import load_dotenv                                # Import dotenv to load variables from .env file

from langchain_google_vertexai import ChatVertexAI            # Import Vertex AI LLM wrapper for Gemini models
from langchain_core.prompts import ChatPromptTemplate         # Import ChatPromptTemplate to create structured prompts
from langchain_core.output_parsers import StrOutputParser     # Import parser to convert AIMessage to plain text
from langchain_core.runnables import RunnableLambda           # Import RunnableLambda to wrap Python functions as runnables

# ==============================
# Load environment variables
# ==============================
load_dotenv()                                                 # Load environment variables from the .env file

required_vars = [                                             # Define required environment variables for Vertex AI
    "GEMINI_MODEL",
    "TEMPERATURE",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_REGION",
    "MAX_OUTPUT_TOKENS"
]

for var in required_vars:                                     # Iterate through required environment variables
    if not os.getenv(var):                                    # Check whether the variable exists
        raise ValueError(f"{var} not found in .env")          # Raise an error if any variable is missing

print("Vertex AI configuration loaded")                       # Confirm Vertex AI configuration is loaded

# ==============================
# TASK 1 — First Prompt
# ==============================
prompt_template = ChatPromptTemplate.from_messages([           # Create the first prompt template
    ("system", "You are a helpful assistant"),                 # System instruction defining assistant behavior
    ("human", "{input}")                                       # Placeholder where user input will be inserted
])

# ==============================
# TASK 2 — LLM
# ==============================
llm_vertexAI = ChatVertexAI(                                   # Initialize Gemini model through Vertex AI
    model_name=os.getenv("GEMINI_MODEL"),                      # Load Gemini model name from environment variables
    temperature=float(os.getenv("TEMPERATURE")),               # Control randomness of the model response
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),                 # Specify Google Cloud project ID
    location=os.getenv("GOOGLE_CLOUD_REGION"),                 # Specify Vertex AI region
    max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS")),     # Set maximum number of tokens the model can generate
    convert_system_message_to_human=True                       # Convert system messages for Gemini compatibility
)

# ==============================
# TASK 3 — Output Parser
# ==============================
str_parser = StrOutputParser()                                 # Convert AIMessage object returned by LLM to plain text

'''
Example:
LLM Output → AIMessage(content="Paris")
Parser Output → "Paris"
'''

# ==============================
# TASK 4 — Custom Runnable
# ==============================
from langchain_core.runnables import RunnableLambda           # Import RunnableLambda to wrap Python functions as runnables

def dictionary_maker(text: str) -> dict:
    """
    Function Description:
    Converts the LLM text output into a dictionary so it can be passed to the next prompt template which expects a {text} variable.
    """
    return {"text": text}                                      # Return dictionary with key 'text'

dictionary_maker_runnable = RunnableLambda(dictionary_maker)  # Wrap Python function into a Runnable for LangChain


# ==============================
# TASK 5 — Second Prompt
# ==============================
prompt_post = ChatPromptTemplate.from_messages([               # Create second prompt template for LinkedIn post generation
    ("system", "You're a social media post generator."),       # System instruction describing assistant role
    ("human", "Create a post for the following text for LinkedIn: {text}")  # Use text generated from previous step
])

# ==============================
# CHAIN PIPELINE
# ==============================
chain = (                                                      # Create a LangChain LCEL pipeline
    prompt_template                                            # Step 1: Generate prompt from user input
    | llm_vertexAI                                             # Step 2: Send prompt to Vertex AI LLM
    | str_parser                                               # Step 3: Convert AIMessage output to plain string
    | dictionary_maker_runnable                                # Step 4: Convert string into dictionary with key 'text'
    | prompt_post                                              # Step 5: Create LinkedIn prompt using dictionary value
    | llm_vertexAI                                             # Step 6: Send LinkedIn prompt to LLM again
    | str_parser                                               # Step 7: Extract final LinkedIn post text
)

result = chain.invoke({                                        # Execute the entire chain pipeline
    "input": "What is the capital of France?"                  # Provide input variable for first prompt
})

print(result)                                                  # Print the final LinkedIn post generated by the chain