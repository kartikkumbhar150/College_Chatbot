# backend/groq_client.py
import os
from groq import Client
from dotenv import load_dotenv

load_dotenv()
print("Groq API Key:", os.getenv("GROQ_API_KEY"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # default to mixtral

if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY in environment (.env)")

client = Client(api_key=GROQ_API_KEY)

def groq_generate(system_prompt, user_prompt, max_tokens=512, temperature=0.0):
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Groq API Error:", str(e))
        return f"[Groq API Error: {str(e)}]"
