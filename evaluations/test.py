import os
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
for i in range(5):
    completion = client.completions.create(model="Meta-Llama-3-70B-Instruct", prompt="San Francisco is a")
    print("Completion result:", completion)