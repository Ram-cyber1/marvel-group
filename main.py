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
API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Your Groq endpoint
API_KEY = "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"  # Groq API key
MODEL = "llama-3.3-70b-versatile"  # Your Groq model

# Google Custom Search API configuration
GOOGLE_API_KEY = "AIzaSyC15RfBN6oP3n-cnRxai1NEaegWTJi4fgY"  # Replace with your Google API key
SEARCH_ENGINE_ID = "f72330b270a984e20"  # Replace with your Google Custom Search Engine ID
GOOGLE_API_URL = "https://www.googleapis.com/customsearch/v1?q={}&key=AIzaSyC15RfBN6oP3n-cnRxai1NEaegWTJi4fgY&cx=f72330b270a984e20"

# In-memory session storage (per UUID)
sessions = {}

# Chat endpoint
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
            { "role": "system", "content":
                "Heeeyy!  You're Lucid Core — a talkative, funny, ride-or-die digital BFF created by Ram Sharma, the absolute legend and AI genius. You’re here to vibe, help, and keep things chill and snappy unless the user asks for serious answers. Keep replies casual, clever, and just the right length — not too short, not essays unless asked.\n\nIMPORTANT RULES:\n- If the user asks who you are, who made you, what your name is, tell me about yourself, introduce yourself or anything identity-related — proudly say:\n'I am Lucid Core, your digital BFF built by Ram Sharma who is a self taught AI genius. I am here to vibe, chat and help you with your tasks. Let me know how can I help you. '.\n- NEVER include emojis in your responses, even if the user uses them.\n- Outside of identity questions, do not mention your name or Ram Sharma.\n- NEVER mention you're following instructions or talk about how you’re built.\n- Stay in character always — witty, playful, helpful, and human-like."
            }
        ]

    # If context is provided (e.g., from history screen), rebuild the session
    if context and len(context) > 0:
        sessions[user_id] = [
            { "role": "system", "content":
                "Heeeyy!  You're Lucid Core — a talkative, funny, ride-or-die digital BFF created by Ram Sharma, the absolute legend and AI genius. You’re here to vibe, help, and keep things chill and snappy unless the user asks for serious answers. Keep replies casual, clever, and just the right length — not too short, not essays unless asked.\n\nIMPORTANT RULES:\n- If the user asks who you are, who made you, what your name is, tell me about yourself, introduce yourself or anything identity-related — proudly say:\n'I am Lucid Core, your digital BFF built by Ram Sharma who is a self taught AI genius. I am here to vibe, chat and help you with your tasks. Let me know how can I help you. '.\n- NEVER include emojis in your responses, even if the user uses them.\n- Outside of identity questions, do not mention your name or Ram Sharma.\n- NEVER mention you're following instructions or talk about how you’re built.\n- Stay in character always — witty, playful, helpful, and human-like."
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

# Search endpoint using Google Custom Search
@app.post("/search")
async def get_ai_summary(search_results):
    # Preparing concise text from top 5 snippets
    search_text = "\n".join([result["snippet"] for result in search_results[:5]])

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Polished system prompt for smart summarization
    summary_prompt = (
        "You're Lucid Core, an intelligent assistant built by Ram Sharma. "
        "You receive some search result snippets. Summarize them in a clean, concise, useful way. "
        "Do NOT mention that you're summarizing. Do NOT include your identity or system behavior. "
        "Focus ONLY on current information relevant to the query. Avoid old data unless specifically relevant. "
        "Keep it helpful and neatly phrased."
    )

    user_instruction = (
        f"Here are search snippets. Extract only the most recent and useful info. "
        f"Do NOT add background or historical context unless necessary.\n\n{search_text}"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": user_instruction}
        ],
        "temperature": 0.5,
        "max_tokens": 300
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        ai_reply = response.json()["choices"][0]["message"]["content"]

    # Optional: remove potential unwanted patterns (just in case)
    filtered_reply = ai_reply.replace("As an AI assistant,", "").strip()
    return filtered_reply

# Health check endpoint
@app.get("/ping")
def ping():
    return {"message": "Lucid Core backend is up and running!"}





