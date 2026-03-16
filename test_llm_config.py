import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_advanced import *
from langfuse.openai import OpenAI as LangfuseOpenAI

def test_llm():
    print(f"Testing LLM with model: {LLM_MODEL}")
    print(f"Base URL: {LLM_BASE_URL}")
    
    client = LangfuseOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Hello, this is a test. Please reply 'OK' if you receive this."}],
            temperature=0,
            max_tokens=10
        )
        content = response.choices[0].message.content
        print(f"Response content: '{content}'")
        if not content:
            print("WARNING: Received empty content from LLM.")
    except Exception as e:
        print(f"LLM Test Error: {e}")

if __name__ == "__main__":
    test_llm()
