from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uuid

app = FastAPI()

# CORS setup for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API Key and model
API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = "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"
MODEL = "mistral-saba-24b"

# Store past messages per user/device UUID
sessions = {}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_msg = data.get("message", "")
    user_id = data.get("uuid", str(uuid.uuid4()))  # use given UUID or generate one

    # If no history exists for this UUID, create new conversation
    if user_id not in sessions:
        sessions[user_id] = [
            {
                "role": "system",
                "content": (
                    "Heeeyy! 😜 I’m Lucid Core, your digital BFF built by Ram Sharma the legend—"
                    "what are we vibin' on today? I'm fun, friendly, and chatty, but I only flex about my creator if you ask 😉"
                )
            }
        ]

    # Add user's message to history
    sessions[user_id].append({"role": "user", "content": user_msg})

    # Prepare and send request to Groq API
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": sessions[user_id],
        "temperature": 0.7,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"]

            # Add Lucid Core's reply to history
            sessions[user_id].append({"role": "assistant", "content": reply})

            return {"reply": reply}

        except Exception as e:
            return {"reply": f"Error occurred: {str(e)}"}
