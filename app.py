import os
import openai
from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-3.5-turbo"

llm = OpenAI(temperature=0.7)

print(llm.predict("What is the weather in the Philippines?"))