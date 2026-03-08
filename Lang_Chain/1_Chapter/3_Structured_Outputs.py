import os, json                                            # Import OS for env access and JSON for parsing model responses
from dotenv import load_dotenv                             # Import dotenv to load environment variables from .env file
from langchain_google_vertexai import ChatVertexAI         # Import Vertex AI LLM wrapper from LangChain
from langchain_core.messages import HumanMessage, SystemMessage  # Import message classes for chat interactions

load_dotenv()                                              # Load variables from .env file into environment

credentials = [                                            # List of required environment variables
    "GEMINI_MODEL",
    "TEMPERATURE",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_REGION",
    "MAX_OUTPUT_TOKENS"
]

for cred in credentials:                                   # Loop through required variables
    if not os.getenv(cred):                                # Check if variable exists in environment
        raise ValueError(f"{cred} is missing in .env file") # Raise error if any required variable is missing

def clean_json(text: str) -> str:
    """
    Function Description:
    Cleans the JSON output returned by the LLM by removing markdown formatting 
    such as " ```json ... ``` " before parsing with json.loads().
    """
    text = text.strip()                                    # Remove leading and trailing whitespace
    if text.startswith("```"):                             # Check if response starts with markdown code block
        text = text.split("```")[1]                         # Extract the inner content between code block markers
        if text.startswith("json"):                         # If the block starts with the word json
            text = text[4:]                                 # Remove the json prefix
    return text.strip()                                     # Return cleaned JSON string


llm_vertexAI = ChatVertexAI(                                # Initialize the Gemini model through Vertex AI
    model_name=os.getenv("GEMINI_MODEL"),                   # Load model name from environment variables
    temperature=float(os.getenv("TEMPERATURE")),            # Control randomness of model output
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),              # Specify Google Cloud project ID
    location=os.getenv("GOOGLE_CLOUD_REGION"),              # Specify Vertex AI region
    max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS")),  # Limit maximum tokens in output
    convert_system_message_to_human=True                    # Convert system messages for compatibility with Gemini
)

# ==========================================================
# Pydantic Structured Output : Principle of the School
# ==========================================================
'''
Pydantic is very strict and gives you run time errors if the key is wrong
'''
from pydantic import BaseModel, Field                       # Import Pydantic base model for schema validation

class JokeSchema(BaseModel):
    """
    Function Description:
    Defines a strict schema for joke output using Pydantic. Ensures runtime validation of keys and data types.
    """
    setup: str = Field(description="The setup for the joke")         # Define setup field with description
    punchline: str = Field(description="The punchline for the joke") # Define punchline field with description

pyd = JokeSchema(                                           # Create example Pydantic object
    setup="some setup",                                     # Provide example setup
    punchline="some punchline"                              # Provide example punchline
)
print(pyd)                                                  # Print validated Pydantic object


def run_pydantic_structured():
    """
    Function Description:
    Generates a joke using the LLM and validates the returned, JSON response against the Pydantic JokeSchema.
    """
    print("\n=== Pydantic Structured Output ===")           # Print section header

    # Define prompt asking for structured JSON response
    prompt = """                                            
    Tell me a joke and return ONLY valid JSON with keys:
    setup, punchline
    At the beginning of the joke, adress me, My name is sriram..!
    """
    response = llm_vertexAI.invoke(prompt)                  # Send prompt to Vertex AI model
    try:                                                    # Attempt JSON parsing
        content = response.content
        cleaned = clean_json(content)                       # Clean markdown formatting from LLM response
        
        data = json.loads(cleaned)                          # Convert JSON string into Python dictionary
        joke = JokeSchema(**data)                           # Validate dictionary against Pydantic schema
        
        print("Setup:", joke.setup)                         # Print validated setup field
        print("Punchline:", joke.punchline)                 # Print validated punchline field
    except Exception as e:                                  # Catch parsing or validation errors
        print("Parsing error:", e)                          
        print("Raw output:", response.content)

# ==========================================================
# TypedDict Structured Output : Class Teacher
# ==========================================================
'''
TypeDict is not very strict and wont gives you run time errors, 
It returns the mistake that you gave and lets you know the mistake that you did..! 
'''
from typing import TypedDict

class JokeSchemaTD(TypedDict):
    """
    Function Description:
    Defines a lightweight schema using TypedDict. Unlike Pydantic, it does not enforce runtime validation.
    """
    setup: str                                               # Setup field expected as string
    punchline: str                                           # Punchline field expected as string

td = JokeSchemaTD({                                          # Create example TypedDict object
    "setup": "setup",                                        # Provide setup value
    "punchline": "punchline"                                 # Provide punchline value
})
print(td)                                                    # Print dictionary output


def run_typeddict_structured():
    """
    Function Description:
    Generates jokes using the LLM and maps the output into a TypedDict structure without strict validation.
    """
    print("\n=== TypedDict Structured Output ===")           # Print section header

    # Define prompt requesting two jokes
    prompt = """                                            
    Tell me two jokes about earth and return ONLY valid JSON with keys:
    setup, punchline
    """
    response = llm_vertexAI.invoke(prompt)                  # Send prompt to LLM
    try:                                                     # Attempt JSON parsing
        content = response.content
        cleaned = clean_json(content)              # Clean markdown formatting
        
        data: JokeSchemaTD = json.loads(cleaned)            # Convert JSON into TypedDict-compatible dictionary
        
        print(data)                                         # Print parsed result
    except Exception as e:                                   # Handle parsing errors
        print("Parsing error:", e)                          # Print error message
        print("Raw output:", response.content)      

# ==========================================================
# MESSAGE BASED INTERACTION: (RAW, PYDANTIC, TYPEDICT)
# ==========================================================
# RAW
def run_message_example():
    """
    Function Description:
    Demonstrates chat-style interaction using SystemMessage and HumanMessage objects with the Vertex AI model.
    """
    print("\n=== Message Example ===")                      # Print section header
    messages = [                                            # Define conversation messages
        SystemMessage(content="You are a Gen-Z assistant who answers in a fun way"),  # System instruction
        HumanMessage(content="Bro tell me two fun fact about earth")                  # User message
    ]
    print(llm_vertexAI.invoke(messages).content)            # Print model response text
    print(llm_vertexAI.invoke(messages).type)               # Print message type returned by model

# PYDANTIC
class EarthFacts(BaseModel):                                # Define Pydantic schema for validation
    fact1: str                                              # First fact field
    fact2: str                                              # Second fact field

def run_message_structured_pydantic():
    """
    Function Description:
    Sends chat-style messages to the LLM and forces the model to return
    structured JSON output which is validated using a Pydantic schema.
    """
    print("\n=== Structured Message Example (Pydantic) ===")       # Print section header
    messages = [                                                  # Define conversation messages
        SystemMessage(content="You are a helpful assistant who returns JSON only"),   # System instruction
        HumanMessage(content="Tell me two fun facts about Earth. Return ONLY valid JSON with keys: fact1, fact2")
    ]
    response = llm_vertexAI.invoke(messages)                      # Send messages to Vertex AI model
    try:
        cleaned = clean_json(response.content)                    # Remove markdown formatting if present
        data = json.loads(cleaned)                                # Convert JSON string to Python dictionary

        facts = EarthFacts(**data)                                # Validate dictionary using Pydantic

        print("Fact 1:", facts.fact1)                             # Print validated fact1
        print("Fact 2:", facts.fact2)                             # Print validated fact2
    except Exception as e:                                        # Handle parsing errors
        print("Parsing error:", e)                                # Print error message
        print("Raw output:", response.content)                    # Print raw model response

# TYPEDICT
class EarthFactsTD(TypedDict):                            # Define TypedDict schema
    fact1: str                                            # First fact field
    fact2: str       

def run_message_structured_typeddict():
    """
    Function Description:
    Sends chat-style messages to the LLM and converts the JSON response
    into a TypedDict structure for lightweight schema typing.
    """
    print("\n=== Structured Message Example (TypedDict) ===")     # Print section header
    messages = [                                                  # Define conversation messages
        SystemMessage(content="You are a helpful assistant who returns JSON only"),   # System instruction
        HumanMessage(content="Tell me two fun facts about Earth. Return ONLY valid JSON with keys: fact1, fact2")
    ]
    response = llm_vertexAI.invoke(messages)                      # Send messages to model
    try:
        cleaned = clean_json(response.content)                    # Clean markdown JSON formatting
        data = json.loads(cleaned)                                # Convert JSON string to dictionary

        facts: EarthFactsTD = data                                # Assign dictionary with TypedDict type hint

        print("Fact 1:", facts["fact1"])                          # Print first fact
        print("Fact 2:", facts["fact2"])                          # Print second fact
    except Exception as e:                                        # Handle parsing errors
        print("Parsing error:", e)                                # Print error message
        print("Raw output:", response.content)                    # Print raw response

if __name__ == "__main__":
    """
    Function Description:
    Entry point of the script. Executes selected demo functions.
    """
    print("Running with Python:", os.sys.executable)         # Display Python interpreter path
    # run_pydantic_structured()                              # Uncomment to test Pydantic structured output
    # run_typeddict_structured()                             # Uncomment to test TypedDict structured output
    # run_message_example()                                  # Uncomment to test message-based interaction
    run_message_structured_pydantic()
    run_message_structured_typeddict()