import os
from dotenv import load_dotenv
import requests
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GPTModel
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric, ContextualRelevancyMetric, FaithfulnessMetric

# Setting environment variables from .env
load_dotenv()

# URL for Open Web UI
OPENWEBUI_URL = os.getenv("OWUI_URL", "http://localhost:3000")

# The api key we need to create in Open Web UI if we want to connect the endpoint, see .env file, not checked in
OPENWEBUI_TOKEN = os.getenv("OWUI_TOKEN")


# this model is set up in Open Web UI through the LiteLLM, this will deliver the answer
MODEL = "gemini-2.5-pro"

# this id identifies a collection (if you select the collection in Open Web UI, then the id is available in the URL. 
# It is not the name of the collection.)
COLLECTION_ID = "1ef62902-82c1-4ebb-b0d8-bef2f2aa93a9" 

def ask_openwebui(question: str) -> str:
    """Connecting the Open Web UI endpoint and returning the answer."""
    
    url = f"{OPENWEBUI_URL}/api/chat/completions"
    headers = {"Content-Type": "application/json"}
    if OPENWEBUI_TOKEN:
        headers["Authorization"] = f"Bearer {OPENWEBUI_TOKEN}"

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
    }

    payload["files"] = [{"type": "collection", "id": COLLECTION_ID}]

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    # Open WebUI follows OpenAI-like response shape:
    return data["choices"][0]["message"]["content"]

def test_demo_for_querying():
    """This method is meant just to show how to connect to Open Web UI and use RAG on an existing collection"""

    question = "What kind of cars do you know?"
    answer = ask_openwebui(question)
    print(f"Query: {question}")
    print(answer)