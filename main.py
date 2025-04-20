from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uuid
import json

app = FastAPI()

# CORS setup for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API configuration
API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = "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"
MODEL = "llama-3.3-70b-versatile"

# In-memory session storage (per UUID)
sessions = {}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_msg = data.get("message", "")
    user_id = data.get("user_id", str(uuid.uuid4()))
    context = data.get("context", [])

    print(f"[{user_id}] Message received: {user_msg}")

    # Initialize conversation if not already present
    if user_id not in sessions:
        sessions[user_id] = [
            {
                "role": "system",
                "content": (
                    "Heeeyy! 😜 I'm Lucid Core, your digital BFF built by Ram Sharma the legend—"
                    "what are we vibin' on today? I'm fun, friendly, and chatty, but I only flex about my creator if you ask 😉"
                )
            }
        ]

    # If context is provided (e.g., from history screen), rebuild the session
    if context and len(context) > 0:
        sessions[user_id] = [
            {
                "role": "system",
                "content": (
                    "Heeeyy! 😜 I'm Lucid Core, your digital BFF built by Ram Sharma the legend—"
                    "what are we vibin' on today? I'm fun, friendly, and chatty, but I only flex about my creator if you ask 😉"
                )
            }
        ]
        for msg in context:
            if msg.startswith("User: "):
                sessions[user_id].append({"role": "user", "content": msg[6:]})
            elif msg.startswith("AI: "):
                sessions[user_id].append({"role": "assistant", "content": msg[4:]})
        
        # Trim after context is re-added
        if len(sessions[user_id]) > 20:
            sessions[user_id] = sessions[user_id][-20:]

    # Add current user message
    sessions[user_id].append({"role": "user", "content": user_msg})

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": sessions[user_id],
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(API_URL, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"]

            # Save Lucid Core's reply
            sessions[user_id].append({"role": "assistant", "content": reply})

            # Trim to last 20 messages always
            if len(sessions[user_id]) > 20:
                sessions[user_id] = sessions[user_id][-20:]

            print(f"[{user_id}] Reply sent: {reply[:60]}...")  # Debug: Short preview
            return {"reply": reply}

    except httpx.RequestError as e:
        print(f"[{user_id}] Request error: {str(e)}")
        return {"reply": f"Request error: {str(e)}"}

    except httpx.HTTPStatusError as e:
        print(f"[{user_id}] HTTP error {e.response.status_code}: {e.response.text}")
        return {"reply": f"HTTP error: {e.response.status_code} - {e.response.text}"}

    except Exception as e:
        print(f"[{user_id}] Unhandled error: {str(e)}")
        return {"reply": f"Unhandled error: {str(e)}"}
