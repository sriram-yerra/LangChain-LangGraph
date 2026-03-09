"""
Script Description:
Creates a LangChain pipeline that:
1. Summarizes a movie using Vertex AI.
2. Converts the summary into a dictionary.
3. Runs two chains in parallel to generate LinkedIn and Instagram posts.
"""

import os                                                     # Import OS module to access environment variables
from dotenv import load_dotenv                                # Import dotenv to load variables from .env file

from langchain_google_vertexai import ChatVertexAI            # Import Vertex AI Gemini chat model wrapper
from langchain_core.prompts import ChatPromptTemplate         # Import prompt template builder for chat models
from langchain_core.output_parsers import StrOutputParser     # Import parser to convert AIMessage → plain string
from langchain_core.runnables import RunnableLambda, RunnableParallel  # Import Runnable utilities for custom functions and parallel execution

# ====================================
# TASK 1 — Load Environment Variables
# ====================================
load_dotenv()                                                 # Load environment variables from the .env file

# ==========================================
# TASK 2 — Validate Vertex AI Configuration
# ==========================================
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

# ==============================
# TASK 3 — Initialize Vertex AI LLM
# ==============================
llm_vertexAI = ChatVertexAI(                                  # Initialize Gemini model through Vertex AI
    model_name=os.getenv("GEMINI_MODEL"),                     # Load model name from environment variables
    temperature=float(os.getenv("TEMPERATURE")),              # Control randomness of generated responses
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),                # Specify Google Cloud project ID
    location=os.getenv("GOOGLE_CLOUD_REGION"),                # Specify Vertex AI region
    max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS")),    # Set maximum number of tokens for model output
    convert_system_message_to_human=True                      # Convert system messages to human messages for Gemini compatibility
)

# ==============================
# TASK 4 — Movie Summary Prompt
# ==============================
prompt_template = ChatPromptTemplate.from_messages([          # Create prompt template for summarizing movies
    ("system", "You are a movie summarizer"),                 # Define system instruction for the model
    ("human", "Please summarize the movie in brief : {input}")# Placeholder where movie name will be inserted
])

# ==============================
# TASK 5 — Output Parser
# ==============================
str_parser = StrOutputParser()                                # Convert AIMessage output from LLM into plain string text

# ===========================================
# TASK 6 — Convert Text → Dictionary Runnable
# ===========================================
def dictionary_maker(text: str) -> dict:
    """
    Function Description:
    Converts LLM text output into a dictionary so it can be
    used by subsequent prompt templates expecting structured input.
    """
    return {"text": text}                                     # Return dictionary with key 'text'

dictionary_maker_runnable = RunnableLambda(dictionary_maker)  # Wrap Python function as Runnable for use inside chain

# ========================================
# TASK 7 — LinkedIn Prompt: LinearChain-1
# ========================================
linkedin_prompt = ChatPromptTemplate.from_messages([                       # Create prompt template for LinkedIn post generation
    ("system", "You are a LinkedIn post generator"),                       # Define role of assistant for LinkedIn content
    ("human", "Create a post for the following text for LinkedIn: {text}") # Use text produced from previous step
])

chain_linkedin = linkedin_prompt | llm_vertexAI | str_parser               # Create chain: Prompt → LLM → Parser for LinkedIn content

# ========================================
# TASK 8 — Instagram Chain: LinearChain-2
# ========================================
insta_prompt = ChatPromptTemplate.from_messages([                           # Create prompt template for Instagram caption
        ("system", "You are an Instagram caption generator"),                   # Define assistant role for Instagram
        ("human", "Create a post for the following text for Instagram: {text}") # Insert summary text into prompt
    ])

chain_instagram = insta_prompt | llm_vertexAI | str_parser               # Create chain: Prompt → LLM → Parser for LinkedIn content

# ----------------------------------------------------------------------------------
'''
This function prepares the data for the next runnable in the chain.
It extracts the required value from the previous step and formats
it into the structure expected by the next prompt template.
'''
def insta_chain(data: dict) -> dict:
    text = data["text"]
    return {"text": text}

insta_chain_runnable = RunnableLambda(insta_chain)

insta_prompt = ChatPromptTemplate.from_messages([                               # Create prompt template for Instagram caption
        ("system", "You are an Instagram caption generator"),                   # Define assistant role for Instagram
        ("human", "Create a post for the following text for Instagram: {text}") # Insert summary text into prompt
    ])

chain_instagram = (
    insta_chain_runnable
    | insta_prompt
    | llm_vertexAI
    | str_parser
)

# ----------------------------------------------------------------------------------
def insta_chain(text: dict):
    """
    Function Description: Generates an Instagram caption using the movie summary text.
    """
    text = text["text"]                                                         # Extract summary text from dictionary input
    insta_prompt = ChatPromptTemplate.from_messages([                           # Create prompt template for Instagram caption
        ("system", "You are an Instagram caption generator"),                   # Define assistant role for Instagram
        ("human", "Create a post for the following text for Instagram: {text}") # Insert summary text into prompt
    ])
    chain_insta = insta_prompt | llm_vertexAI | str_parser                      # Create Instagram generation chain
    result = chain_insta.invoke({"text": text})
    return result                                                               # Execute chain and return generated caption

insta_chain_runnable = RunnableLambda(insta_chain)                              # Wrap Instagram function as Runnable

# ----------------------------------------------------------------------------------
def insta_chain(data: dict):
    """
    Function Description:
    Extracts the movie summary text and creates the Instagram
    prompt that will later be processed by the LLM.
    """
    text = data["text"]                                                         # Extract summary text from dictionary input
    insta_prompt = ChatPromptTemplate.from_messages([                           # Create prompt template for Instagram caption
        ("system", "You are an Instagram caption generator"),                   # Define assistant role for Instagram
        ("human", "Create a post for the following text for Instagram: {text}") # Insert summary text into prompt
    ])
    '''
    This is not a prompt string yet, It is a template with variables.
    ".invoke()" fills the template variables with actual values and produces the final prompt.
    '''
    result = insta_prompt.invoke({"text": text})
    return result                                                               # Return formatted prompt messages (ChatPromptValue)

insta_chain_runnable = RunnableLambda(insta_chain)                              # Convert the function into a Runnable

chain_instagram = insta_chain_runnable | llm_vertexAI | str_parser     

# ----------------------------------------------------------------------------------
'''
Example, Suppose:
text = "KGF is a story about Rocky rising from poverty..."

After .invoke() the prompt becomes:
System: You are an Instagram caption generator
Human: Create a post for the following text for Instagram: KGF is a story about Rocky rising from poverty...

The returned object is a ChatPromptValue, which contains the formatted messages.
ChatPromptValue(
    messages=[
        SystemMessage(...),
        HumanMessage(...)
    ]
)
'''

# ==================================================
# TASK 9 — Final Parallel Chain: Final Orchestration
# ==================================================
final_chain = (                                                           # Build full pipeline
    prompt_template                                         # Step 1: Generate movie summary prompt
    | llm_vertexAI                                          # Step 2: Send prompt to Vertex AI model
    | str_parser                                            # Step 3: Convert AIMessage output into string
    | dictionary_maker_runnable                             # Step 4: Convert summary string into dictionary
    | RunnableParallel(                                     # Step 5: Execute multiple chains in parallel
        branches={                                          # Define parallel branches
            "linkedin": chain_linkedin,                     # Branch 1: Generate LinkedIn post
            "instagram": chain_instagram                    # Branch 2: Generate Instagram caption
        }
    )
)
'''
Whatever we create in chain, it is runnable by default..!
'''

# ==============================
# TASK 10 — Run the Pipeline
# ==============================
result = final_chain.invoke({"input": "KGF"})                  # Run the full pipeline with movie name "KGF"
print(result)                                                  # Print results from both parallel branches

# ================================
# LAST TASK — Beautify Function
# ================================
def beautify(final_response: dict) -> dict:
    """
    Function Description:
    Extracts LinkedIn and Instagram responses from the parallel
    chain output and returns a simplified dictionary.
    """
    linkedin_response = final_response['branches']['linkedin']      # Extract LinkedIn output from the parallel chain result
    instagram_response = final_response['branches']['instagram']    # Extract Instagram output from the parallel chain result

    return {"linkedin": linkedin_response, "instagram": instagram_response}  # Return clean dictionary with both outputs

beautify_runnable = RunnableLambda(beautify)                        # Convert the Python function into a Runnable component

# ===========================
# TASK 2 — Beautified Chain
# ===========================
beautified_chain = final_chain | beautify_runnable                  # Extend the existing pipeline by adding the beautify step

result = beautified_chain.invoke("Pushpa")                          # Execute the full pipeline using movie name "Pushpa"

print(result)                                                       # Print cleaned output