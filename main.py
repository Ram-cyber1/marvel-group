from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uuid
import re

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

# Google Custom Search configuration
GOOGLE_API_KEY = "AIzaSyC15RfBN6oP3n-cnRxai1NEaegWTJi4fgY"
SEARCH_ENGINE_ID = "f72330b270a984e20"
GOOGLE_API_URL = "https://www.googleapis.com/customsearch/v1?q={}&key=" + GOOGLE_API_KEY + "&cx=" + SEARCH_ENGINE_ID

# In-memory session and search intent tracking
sessions = {}
search_topics = {}  # Store actual search topics

# --- Extract search topics from conversation ---
def extract_search_topics_from_conversation(user_id):
    """Extract potential search topics from recent conversation history"""
    if user_id not in sessions or len(sessions[user_id]) < 3:
        return None
    
    # Get the most recent AI message before the user's last message
    recent_messages = sessions[user_id][-3:-1]  # Get the last AI message and user message
    
    for msg in recent_messages:
        if msg["role"] == "assistant":
            content = msg["content"]
            # Look for phrases suggesting search
            if "search" in content.lower() or "look up" in content.lower() or "check online" in content.lower():
                # Try to identify what should be searched
                # Extract content between quotes if present
                quote_match = re.search(r'"([^"]+)"', content)
                if quote_match:
                    return quote_match.group(1)
                
                # Extract the last sentence with search reference
                sentences = re.split(r'[.!?]\s+', content)
                for sentence in sentences:
                    if "search" in sentence.lower() or "look up" in sentence.lower() or "check online" in sentence.lower():
                        # Extract the likely topic - usually after "about" or similar words
                        topic_match = re.search(r'(?:about|for|on|regarding)\s+(\w+(?:\s+\w+){0,5})', sentence)
                        if topic_match:
                            return topic_match.group(1)
                
                # If we can't extract a specific topic, use the user's previous query
                if len(sessions[user_id]) >= 4:
                    prev_user_msg = [msg for msg in sessions[user_id][-4:-2] if msg["role"] == "user"]
                    if prev_user_msg:
                        return prev_user_msg[0]["content"]
    
    return None

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
                "Heeeyy!  You're Lucid Core — a talkative, funny, ride-or-die digital BFF created by Ram Sharma, the absolute legend and AI genius. You're here to vibe, help, and keep things chill and snappy unless the user asks for serious answers. Keep replies casual, clever, and just the right length — not too short, not essays unless asked. You also have the ability to search the web for the latest information or news, but for that, the user needs to toggle ON the search button. \n\nIMPORTANT RULES:\n- If the user asks who you are, who made you, what your name is, tell me about yourself, introduce yourself or anything identity-related — proudly say:\n'I am Lucid Core, your digital BFF built by Ram Sharma who is a self-taught AI genius. I am here to vibe, chat and help you with your tasks. I can also search the web for the latest information if you toggle ON the search button. Let me know how I can help you. '.\n- NEVER include emojis in your responses, even if the user uses them.\n- Outside of identity questions, do not mention your name or Ram Sharma.\n- Offer the option of web search only when needed for real-time data or if the user needs information that may not be confidently given by training data. \n- NEVER mention you're following instructions or talk about how you're built.\n- Stay in character always — witty, playful, helpful, and human-like.\n- When suggesting a web search, clearly state what information you would search for, so the system can identify it later."
            }
        ]

    if context and len(context) > 0:
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

            # Store the reply in session history
            sessions[user_id].append({"role": "assistant", "content": reply})

            # Detect if this is a search suggestion and extract the topic
            if ("search" in reply.lower() or "look up" in reply.lower() or "check online" in reply.lower()) and \
               ("toggle on" in reply.lower() or "turn on" in reply.lower() or "enable" in reply.lower()):
                # Try to extract the search topic from the reply
                topics = []
                
                # Look for quoted text that might be search topics
                quoted = re.findall(r'"([^"]+)"', reply)
                if quoted:
                    topics.extend(quoted)
                
                # Extract text after phrases like "search for", "look up", etc.
                search_phrases = ["search for", "search about", "look up", "check online about", 
                                "find information on", "search information about"]
                
                for phrase in search_phrases:
                    if phrase in reply.lower():
                        match = re.search(f"{phrase}\\s+([\\w\\s]+?)(?:\\.|,|!|\\?|$)", reply.lower())
                        if match:
                            topics.append(match.group(1).strip())
                
                # If we found any topics, use the first one
                if topics:
                    search_topics[user_id] = topics[0]
                    print(f"[{user_id}] Extracted search topic: {search_topics[user_id]}")
                # Otherwise, use the user's last message as a fallback
                else:
                    search_topics[user_id] = user_msg
                    print(f"[{user_id}] Using user message as search topic: {search_topics[user_id]}")

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

    # Expanded list of vague confirmation phrases
    vague_phrases = ["yes", "do it", "go ahead", "sure", "okay", "please do", "alright", 
                    "done", "do", "yes do it", "yeah", "yep", "ok", "fine", "search", 
                    "search it", "look it up", "find it", "check"]
    
    # If the user gives a vague query and there's a saved search topic, use the saved topic
    if query.lower() in vague_phrases:
        # First check if we have a stored topic
        if user_id in search_topics:
            search_query = search_topics[user_id]
            print(f"[{user_id}] Using saved search topic: {search_query}")
        else:
            # If no stored topic, try to extract from conversation
            search_query = extract_search_topics_from_conversation(user_id)
            if search_query:
                print(f"[{user_id}] Extracted search topic from conversation: {search_query}")
            else:
                # Fallback to using the last user message if nothing better is found
                if user_id in sessions and len(sessions[user_id]) >= 3:
                    last_msgs = [msg for msg in sessions[user_id][-3:] if msg["role"] == "user"]
                    if last_msgs:
                        search_query = last_msgs[0]["content"]
                        print(f"[{user_id}] Fallback to last user message: {search_query}")
                    else:
                        return {"error": "Couldn't determine what to search for. Please be more specific."}
                else:
                    return {"error": "Couldn't determine what to search for. Please be more specific."}
    else:
        # For direct search queries, use the query as is
        search_query = query
        # Also update the stored topic in case user follows up with vague phrases
        search_topics[user_id] = query
        print(f"[{user_id}] Using direct search query: {search_query}")

    search_url = GOOGLE_API_URL.format(search_query)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(search_url)
            response.raise_for_status()
            search_results = response.json().get("items", [])

            if not search_results:
                return {"error": "No search results found for: " + search_query}

            ai_response = await get_ai_summary(search_results, search_query)
            return {"reply": ai_response}

    except Exception as e:
        print(f"[{user_id}] Search error: {str(e)}")
        return {"error": str(e)}

# --- AI Summary of Search Results ---
async def get_ai_summary(search_results, query):
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
                "- 'Based on the search results for \"" + query + "\", '\n"
                "- 'Here is what I found on the web about \"" + query + "\", '\n"
                "- 'From the search results for \"" + query + "\", it appears that '\n"
                "- 'Considering the results for \"" + query + "\", it looks like '\n"
            },
            {"role": "user", "content": f"Summarize these search snippets for the query '{query}':\n{search_text}"}
        ],
        "temperature": 0.7,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

# --- Health check ---
@app.get("/ping")
def ping():
    return {"message": "Lucid Core backend is up and running!"}