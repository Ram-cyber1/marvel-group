from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uuid
import re
import base64
import io
from PIL import Image
from typing import List, Dict, Any, Optional

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

# Hugging Face API configuration
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/"
import os

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


# Hugging Face model endpoints
IMAGE_CAPTIONING_MODEL = "Salesforce/blip-image-captioning-base"
IMAGE_GENERATION_MODEL = "stabilityai/stable-diffusion-2"
OCR_MODEL = "microsoft/trocr-base-printed"

# In-memory session and search tracking
sessions = {}
search_contexts = {}  # Enhanced structure to store search contexts

# --- Advanced search context extraction ---
def extract_search_context(text: str) -> Optional[str]:
    """Extract potential search topics from AI response with multiple methods"""
    text = text.lower()
    search_context = None
    
    # Method 1: Look for quoted text near search suggestions
    quoted_matches = re.findall(r'"([^"]+)"', text)
    if quoted_matches:
        # Find quotes near search terms
        search_terms = ["search", "look up", "find", "check online"]
        for term in search_terms:
            for match in quoted_matches:
                if term in text.split('"' + match + '"')[0][-30:] or \
                   term in text.split('"' + match + '"')[1][:30]:
                    return match
    
    # Method 2: Extract phrases after common search suggestion patterns
    patterns = [
        r"search (?:for|about)\s+([^.,!?]+)",
        r"look up\s+([^.,!?]+)",
        r"find (?:info|information) (?:about|on)\s+([^.,!?]+)",
        r"check online (?:for|about)\s+([^.,!?]+)",
        r"toggle on (?:to search|and search) (?:for|about)\s+([^.,!?]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            search_context = match.group(1).strip()
            break
    
    # Method 3: If AI suggests toggle ON search, look for the most relevant noun phrases
    if not search_context and ("toggle on" in text or "turn on" in text) and \
       ("search" in text or "look up" in text):
        # Look for the sentence with the toggle suggestion
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            if "toggle on" in sentence.lower() or "turn on" in sentence.lower():
                # Extract noun phrases after key terms
                noun_match = re.search(r'(?:about|for|on|regarding)\s+(\w+(?:\s+\w+){0,5})', sentence)
                if noun_match:
                    search_context = noun_match.group(1).strip()
                    break
    
    return search_context

def build_search_context_from_history(user_id: str) -> Dict[str, Any]:
    """Build comprehensive search context from conversation history"""
    if user_id not in sessions or len(sessions[user_id]) < 2:
        return {"primary_topic": None, "related_terms": [], "recent_messages": []}
    
    context = {
        "primary_topic": None,
        "related_terms": [],
        "recent_messages": []
    }
    
    # Collect recent messages for context
    recent_messages = sessions[user_id][-6:]  # Get last 6 messages for context
    user_messages = [msg["content"] for msg in recent_messages if msg["role"] == "user"]
    ai_messages = [msg["content"] for msg in recent_messages if msg["role"] == "assistant"]
    
    context["recent_messages"] = user_messages + ai_messages
    
    # Extract search topic from most recent AI message
    if ai_messages:
        last_ai_msg = ai_messages[-1]
        search_topic = extract_search_context(last_ai_msg)
        if search_topic:
            context["primary_topic"] = search_topic
            
            # Also extract related terms from user messages
            if user_messages:
                # Extract keywords from most recent user message
                last_user_msg = user_messages[-1]
                # Simple keyword extraction - get nouns and noun phrases
                words = re.findall(r'\b[A-Za-z]{3,}\b', last_user_msg)
                context["related_terms"] = [w for w in words if w.lower() not in 
                                          ['the', 'and', 'for', 'that', 'what', 'when', 'where', 'how', 'why']][:5]
    
    # If no topic found from AI message but user has a recent message, use it as fallback
    if not context["primary_topic"] and user_messages:
        context["primary_topic"] = user_messages[-1]
    
    return context

# --- Chat Endpoint with Enhanced Context Awareness ---
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

            # Process reply to detect search suggestions
            has_search_suggestion = ("toggle on" in reply.lower() or "turn on" in reply.lower()) and \
                                 ("search" in reply.lower() or "look up" in reply.lower())

            # Extract search topic if there's a suggestion
            if has_search_suggestion:
                # Extract the search context
                search_topic = extract_search_context(reply)
                
                if search_topic:
                    # Store comprehensive search context
                    context_data = build_search_context_from_history(user_id)
                    context_data["primary_topic"] = search_topic  # Override with extracted topic
                    search_contexts[user_id] = context_data
                    print(f"[{user_id}] Extracted search context: {search_topic}")

            # Store the reply in session history
            sessions[user_id].append({"role": "assistant", "content": reply})

            if len(sessions[user_id]) > 20:
                sessions[user_id] = sessions[user_id][-20:]

            print(f"[{user_id}] Reply sent: {reply[:60]}...")
            return {"reply": reply}

    except Exception as e:
        print(f"[{user_id}] Error: {str(e)}")
        return {"reply": f"Error: {str(e)}"}

# --- Improved Search Endpoint ---
@app.post("/search")
async def search(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()
    user_id = data.get("user_id", None)

    print(f"[{user_id}] Incoming search query: '{query}'")

    if not user_id:
        return {"error": "Missing user ID. Cannot retrieve intent context."}

    # List of vague confirmation phrases
    vague_phrases = ["yes", "do it", "go ahead", "sure", "okay", "please do", "alright", 
                    "done", "do", "yes do it", "yeah", "yep", "ok", "fine", "search", 
                    "search it", "look it up", "find it", "check", "toggle on", "enable search"]
    
    # Determine search query based on context and user input
    search_query = query
    
    # If the user gives a vague confirmation, use stored context
    if query.lower() in vague_phrases:
        if user_id in search_contexts and search_contexts[user_id]["primary_topic"]:
            search_query = search_contexts[user_id]["primary_topic"]
            print(f"[{user_id}] Using stored search context: {search_query}")
        else:
            # If no stored context, build it from conversation history
            context_data = build_search_context_from_history(user_id)
            if context_data["primary_topic"]:
                search_query = context_data["primary_topic"]
                search_contexts[user_id] = context_data
                print(f"[{user_id}] Built search context from history: {search_query}")
            else:
                # Last resort: use the last user message
                if user_id in sessions and len(sessions[user_id]) >= 3:
                    user_msgs = [msg for msg in sessions[user_id][-5:] if msg["role"] == "user"]
                    if user_msgs:
                        search_query = user_msgs[0]["content"]
                        print(f"[{user_id}] Fallback to user message: {search_query}")
                    else:
                        return {"error": "Couldn't determine what to search for. Please be more specific."}
                else:
                    return {"error": "Couldn't determine what to search for. Please be more specific."}
    else:
        # Direct search query - update context with this query
        if user_id not in search_contexts:
            search_contexts[user_id] = build_search_context_from_history(user_id)
        search_contexts[user_id]["primary_topic"] = query
        print(f"[{user_id}] Using direct search query and updating context: {query}")

    # Execute the search
    search_url = GOOGLE_API_URL.format(search_query)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(search_url)
            response.raise_for_status()
            search_results = response.json().get("items", [])

            if not search_results:
                return {"error": "No search results found for: " + search_query}

            # Get AI to summarize with enhanced context
            ai_response = await get_enhanced_ai_summary(search_results, search_query, user_id)
            
            # Add the search response to the session history
            if user_id in sessions:
                sessions[user_id].append({"role": "user", "content": f"[SEARCH QUERY: {search_query}]"})
                sessions[user_id].append({"role": "assistant", "content": ai_response})
            
            return {"reply": ai_response}

    except Exception as e:
        print(f"[{user_id}] Search error: {str(e)}")
        return {"error": str(e)}

# --- Enhanced AI Summary with Context Awareness ---
async def get_enhanced_ai_summary(search_results, query, user_id):
    # Prepare search snippets
    search_text = "\n".join([f"Title: {item.get('title', '')}\nURL: {item.get('link', '')}\nDescription: {item.get('snippet', '')}" 
                          for item in search_results[:5]])
    
    # Get conversation context
    context_prompt = ""
    if user_id in sessions and len(sessions[user_id]) > 1:
        # Get last few exchanges
        recent_messages = sessions[user_id][-6:]  # Last 6 messages
        conversation_summary = "\n".join([f"{msg['role'].capitalize()}: {msg['content'][:100]}..." 
                                       for msg in recent_messages])
        context_prompt = f"\nRecent conversation context:\n{conversation_summary}\n\n"
    
    # Add related terms if available
    related_terms = ""
    if user_id in search_contexts and search_contexts[user_id]["related_terms"]:
        terms = search_contexts[user_id]["related_terms"]
        related_terms = f"Related terms from conversation: {', '.join(terms)}\n\n"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = (
        "You are Lucid Core, a talkative, funny, ride-or-die digital BFF. Keep responses casual, clever, and just the "
        "right length. Maintain this personality when summarizing search results. Be informative but with a friendly, "
        "slightly playful tone. Start your search result summaries in one of these ways (choose one that fits best):\n"
        f"- 'Based on my search for \"{query}\", here's what I found:'\n"
        f"- 'Just checked the web for \"{query}\" and here's the scoop:'\n"
        f"- 'Here's what the internet says about \"{query}\":'\n"
        f"- 'So I looked up \"{query}\" and found this:'\n\n"
        "Highlight the most important and recent information. Avoid mentioning that you're an AI or following instructions."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context_prompt}{related_terms}Summarize these search results for the query '{query}':\n\n{search_text}"}
        ],
        "temperature": 0.7,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

# --- Image Analysis Endpoint (BLIP Image Captioning) ---
@app.post("/image-analysis")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an image using Salesforce/blip-image-captioning-base model
    """
    try:
        # Validate HUGGINGFACE_API_KEY
        if not HUGGINGFACE_API_KEY:
            return {"error": "Hugging Face API key not configured"}
            
        # Validate file content type
        content_type = file.content_type
        if not content_type or not content_type.startswith('image/'):
            return {"error": "Uploaded file must be an image"}
        
        # Read image file
        image_content = await file.read()
        
        # Validate file size (Max 10MB)
        if len(image_content) > 10 * 1024 * 1024:
            return {"error": "Image size exceeds the limit (max 10MB)"}
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Encode image as base64
        encoded_image = base64.b64encode(image_content).decode("utf-8")
        
        # Send request to Hugging Face API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{HUGGINGFACE_API_URL}{IMAGE_CAPTIONING_MODEL}",
                headers=headers,
                json={"inputs": {"image": encoded_image}},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            caption = ""
            if isinstance(result, list) and len(result) > 0:
                caption = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                caption = result.get("generated_text", "")
            else:
                caption = str(result)
                
            print(f"Image analysis result: {caption[:50]}...")
            return {"caption": caption, "success": True}
    
    except httpx.HTTPStatusError as e:
        print(f"Image analysis HTTP error: {str(e)}")
        return {"error": f"API error: {e.response.status_code}", "success": False}
    except Exception as e:
        print(f"Image analysis error: {str(e)}")
        return {"error": f"Failed to analyze image: {str(e)}", "success": False}

# --- Image Generation Endpoint (Stable Diffusion) ---
@app.post("/image-generation")
async def generate_image(request: Request):
    """
    Generate an image using stabilityai/stable-diffusion-2 model
    """
    try:
        # Validate HUGGINGFACE_API_KEY
        if not HUGGINGFACE_API_KEY:
            return {"error": "Hugging Face API key not configured", "success": False}
            
        data = await request.json()
        prompt = data.get("prompt", "")
        
        if not prompt:
            return {"error": "Prompt is required for image generation", "success": False}
        
        # Ensure prompt is not too long (typical limit is around 75-100 tokens)
        if len(prompt) > 500:
            prompt = prompt[:500]
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        }
        
        # For Stable Diffusion, we need to handle binary response
        async with httpx.AsyncClient() as client:
            # We're expecting binary data (image) back
            response = await client.post(
                f"{HUGGINGFACE_API_URL}{IMAGE_GENERATION_MODEL}",
                headers=headers,
                json={"inputs": prompt},
                timeout=180  # Longer timeout for image generation
            )
            response.raise_for_status()
            
            # If we got here, we have our image data
            image_bytes = response.content
            
            # Convert to base64 for response (frontend expects base64 for display)
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            
            print(f"Generated image for prompt: {prompt[:30]}...")
            return {
                "image": encoded_image,
                "format": "base64",
                "prompt": prompt,
                "success": True
            }
    
    except httpx.HTTPStatusError as e:
        print(f"Image generation HTTP error: {str(e)}")
        return {"error": f"API error: {e.response.status_code}", "success": False}
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        return {"error": f"Failed to generate image: {str(e)}", "success": False}

# --- OCR Endpoint (TrOCR) ---
@app.post("/image-ocr")
async def process_ocr(file: UploadFile = File(...)):
    """
    Extract text from images using microsoft/trocr-base-printed model
    """
    try:
        # Validate HUGGINGFACE_API_KEY
        if not HUGGINGFACE_API_KEY:
            return {"error": "Hugging Face API key not configured", "success": False}
            
        # Validate file content type
        content_type = file.content_type
        if not content_type or not content_type.startswith('image/'):
            return {"error": "Uploaded file must be an image", "success": False}
        
        # Read image file
        image_content = await file.read()
        
        # Validate file size (Max 10MB)
        if len(image_content) > 10 * 1024 * 1024:
            return {"error": "Image size exceeds the limit (max 10MB)", "success": False}
        
        # Try to optimize image for OCR if PIL is available
        try:
            img = Image.open(io.BytesIO(image_content))
            # Convert to RGB if not already (handles RGBA, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize if too large (helps with API limits and processing speed)
            max_dim = 1000
            if max(img.width, img.height) > max_dim:
                ratio = max_dim / max(img.width, img.height)
                new_width = int(img.width * ratio)
                new_height = int(img.height * ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save to bytes
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            image_content = buffer.getvalue()
        except Exception as img_error:
            print(f"Image optimization skipped: {str(img_error)}")
            # Continue with original image if optimization fails
            pass
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Encode image as base64
        encoded_image = base64.b64encode(image_content).decode("utf-8")
        
        # Send request to Hugging Face API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{HUGGINGFACE_API_URL}{OCR_MODEL}",
                headers=headers,
                json={"inputs": {"image": encoded_image}},
                timeout=90
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract text safely handling different response formats
            extracted_text = ""
            if isinstance(result, list) and len(result) > 0:
                extracted_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                extracted_text = result.get("generated_text", "")
            else:
                extracted_text = str(result)
                
            print(f"OCR result: {extracted_text[:50]}...")
            return {"text": extracted_text, "success": True}
    
    except httpx.HTTPStatusError as e:
        print(f"OCR HTTP error: {str(e)}")
        return {"error": f"API error: {e.response.status_code}", "success": False}
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return {"error": f"Failed to extract text from image: {str(e)}", "success": False}

# --- Health check ---
@app.get("/ping")
def ping():
    return {"message": "Lucid Core backend is up and running!"}