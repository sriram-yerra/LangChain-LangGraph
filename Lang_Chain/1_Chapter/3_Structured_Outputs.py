import os
import json
from dotenv import load_dotenv

# LangChain + Vertex AI
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage

# Pydantic schema
from pydantic import BaseModel, Field

# TypedDict schema
from typing import TypedDict


# -----------------------------
# Load environment variables
# -----------------------------
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
        raise ValueError(f"{var} is missing in .env file")


# -----------------------------
# Initialize Vertex AI LLM
# -----------------------------
llm_vertexAI = ChatVertexAI(
    model_name=os.getenv("GEMINI_MODEL"),          # gemini-1.5-flash
    temperature=float(os.getenv("TEMPERATURE")),
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_REGION"),
    max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS"))
)


# ==========================================================
# Pydantic Structured Output (manual JSON parsing)
# ==========================================================
class JokeSchema(BaseModel):
    setup: str = Field(description="The setup for the joke")
    punchline: str = Field(description="The punchline for the joke")


def run_pydantic_structured():
    print("\n=== Pydantic Structured Output ===")

    prompt = """
    Tell me a joke and return ONLY valid JSON with keys:
    setup, punchline

    Example:
    {
        "setup": "...",
        "punchline": "..."
    }
    """

    response = llm_vertexAI.invoke(prompt)

    try:
        data = json.loads(response.content)
        joke = JokeSchema(**data)

        print("Setup:", joke.setup)
        print("Punchline:", joke.punchline)

    except Exception as e:
        print("Parsing error:", e)
        print("Raw output:", response.content)


# ==========================================================
# TypedDict Structured Output (manual parsing)
# ==========================================================
class JokeSchemaTD(TypedDict):
    setup: str
    punchline: str


def run_typeddict_structured():
    print("\n=== TypedDict Structured Output ===")

    prompt = """
    Tell me a joke and return ONLY valid JSON with keys:
    setup, punchline
    """

    response = llm_vertexAI.invoke(prompt)

    try:
        data: JokeSchemaTD = json.loads(response.content)

        print(data)

    except Exception as e:
        print("Parsing error:", e)
        print("Raw output:", response.content)


# ==========================================================
# Message based conversation
# ==========================================================
def run_message_example():
    print("\n=== Message Example ===")

    messages = [
        SystemMessage(content="You are a Gen-Z assistant who answers in a fun way"),
        HumanMessage(content="Bro tell me a fun fact")
    ]

    response = llm_vertexAI.invoke(messages)

    print(response.content)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    print("Running with Python:", os.sys.executable)

    run_pydantic_structured()
    run_typeddict_structured()
    run_message_example()