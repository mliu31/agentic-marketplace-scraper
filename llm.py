from google import genai
from dotenv import load_dotenv

load_dotenv()

system_prompt = "You are an AI agent that will automate scraping online marketplaces for specific products. "

def query_gemini(question: str) -> str:
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=question,
        system_instruction=system_prompt
    )
    return response.text

# print(query_gemini("hi gemi"))