import os
from openai import OpenAI
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    temperature=0
)

multilingual_llm = ChatOpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    model="Qwen/Qwen3-32B",
    temperature=0
)

super_basic_model = ChatOpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    model="meta-llama/Llama-3.2-1B-Instruct",
    temperature=0
)

standard_model = ChatOpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    temperature=0
)

larger_context_model = ChatOpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    temperature=0
)