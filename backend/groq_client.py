import os
from groq import Client
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY in environment (.env)")

client = Client(api_key=GROQ_API_KEY)


def groq_generate(system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Generate a completion from Groq. Raises Exception on failure."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    try:
        choice = response.choices[0]
        msg = getattr(choice, "message", None)

        if msg:
            content = getattr(msg, "content", None)
            if not content and isinstance(msg, dict):
                content = msg.get("content")
        else:
            if isinstance(choice, dict):
                content = (choice.get("message") or {}).get("content")
            else:
                content = None

        if not content:
            raise RuntimeError("No content returned from Groq response")

        return content.strip()
    except Exception as e:
        raise RuntimeError(f"Groq generation failed: {e}")
