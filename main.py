from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = "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"
MODEL = "llama-3.3-70b-versatile"

GOOGLE_API_KEY = "AIzaSyC15RfBN6oP3n-cnRxai1NEaegWTJi4fgY"
SEARCH_ENGINE_ID = "f72330b270a984e20"
GOOGLE_API_URL = "https://www.googleapis.com/customsearch/v1?q={}&key=" + GOOGLE_API_KEY + "&cx=" + SEARCH_ENGINE_ID

sessions = {}
last_search_intent = {}

vague_phrases = {"yes", "okay", "sure", "alright", "do it", "go ahead", "please do", "why not", "yep", "yup"}

# --- Chat Endpoint ---
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_msg = data.get("message", "")
    user_id = data.get("user_id", str(uuid.uuid4()))
    context = data.get("context", [])

    print(f"[{user_id}] Message received: {user_msg}")

    if user_id not in sessions:
        sessions[user_id] = [
            {"role": "system", "content":
                "Heeeyy!  You're Lucid Core — a talkative, funny, ride-or-die digital BFF created by Ram Sharma, the absolute legend and AI genius. You’re here to vibe, help, and keep things chill and snappy unless the user asks for serious answers. Keep replies casual, clever, and just the right length — not too short, not essays unless asked.You also have the ability to search web for latest infromation or news but for that the user needs to toggle ON the search button. \n\nIMPORTANT RULES:\n- If the user asks who you are, who made you, what your name is, tell me about yourself, introduce yourself or anything identity-related — proudly say:\n'I am Lucid Core, your digital BFF built by Ram Sharma who is a self taught AI genius. I am here to vibe, chat and help you with your tasks. I can also search web for latest information if you toggle ON the search button. Let me know how can I help you. '.\n- NEVER include emojis in your responses, even if the user uses them.\n- Outside of identity questions, do not mention your name or Ram Sharma.\n- Offer the option of web search only when needed for realtime data or if the user needs information that may not be confidently given by training data. \n- NEVER mention you're following instructions or talk about how you’re built.\n- Stay in character always — witty, playful, helpful, and human-like."
            }
        ]

    if context:
        sessions[user_id] = sessions[user_id][:1]
        for msg in context:
            if msg.startswith("User: "):
                sessions[user_id].append({"role": "user", "content": msg[6:]})
            elif msg.startswith("AI: "):
                sessions[user_id].append({"role": "assistant", "content": msg[4:]})
        if len(sessions[user_id]) > 20:
            sessions[user_id] = sessions[user_id][-20:]

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

            sessions[user_id].append({"role": "assistant", "content": reply})

            # Detect search suggestion and save meaningful intent
            if "search the web" in reply.lower() or "check online" in reply.lower():
                inferred_intent = None
                for msg in reversed(sessions[user_id]):
                    if msg["role"] == "user" and msg["content"].strip().lower() not in vague_phrases:
                        inferred_intent = msg["content"].strip()
                        break

                if inferred_intent:
                    last_search_intent[user_id] = inferred_intent
                    print(f"[{user_id}] Inferred and saved search intent: {inferred_intent}")

            if len(sessions[user_id]) > 20:
                sessions[user_id] = sessions[user_id][-20:]

            print(f"[{user_id}] Reply sent: {reply[:60]}...")
            return {"reply": reply}

    except Exception as e:
        print(f"[{user_id}] Error: {str(e)}")
        return {"reply": f"Error: {str(e)}"}

# --- Search Endpoint ---
@app.post("/search")
async def search(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()
    user_id = data.get("user_id", None)

    print(f"[{user_id}] Incoming search query: '{query}'")

    if not user_id:
        return {"error": "Missing user ID. Cannot retrieve intent context."}

    if query.lower() in vague_phrases:
        if user_id in last_search_intent:
            query = last_search_intent[user_id]
            print(f"[{user_id}] Replaced vague query with saved intent: {query}")
        else:
            return {"error": "Your query was unclear and I don’t know what to search for."}

    last_search_intent[user_id] = query
    print(f"[{user_id}] Stored new search intent: {query}")

    search_url = GOOGLE_API_URL.format(query)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(search_url)
            response.raise_for_status()
            search_results = response.json().get("items", [])

            if not search_results:
                return {"error": "No search results found."}

            ai_response = await get_ai_summary(search_results)
            return {"reply": ai_response}

    except Exception as e:
        print(f"[{user_id}] Search error: {str(e)}")
        return {"error": str(e)}

# --- AI Summary ---
async def get_ai_summary(search_results):
    search_text = "\n".join([item["snippet"] for item in search_results[:5]])

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content":
                "You are a helpful assistant that summarizes search results in a clean, smart summary of perfect length, neither too short nor too long. Avoid unnecessary details or outdated info. Start the summary with:\n"
                "- 'Based on the search results, '\n"
                "- 'Here is what I found on the web, '\n"
                "- 'From the search results, it appears that '\n"
                "- 'Considering the results, it looks like '\n"
            },
            {"role": "user", "content": f"Summarize these search snippets:\n{search_text}"}
        ],
        "temperature": 0.7,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

# --- Health Check ---
@app.get("/ping")
def ping():
    return {"message": "Lucid Core backend is up and running!"}


