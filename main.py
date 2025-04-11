from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import groq
import os

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# CORS middleware to allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# For incoming user messages
class Message(BaseModel):
    message: str

# Store chat context in memory
session = {
    "used_intro": False,
    "history": []
}

# POST route for chat
@app.post("/chat")
async def chat(msg: Message):
    user_message = msg.message
    session["history"].append({"role": "user", "content": user_message})

    if not session["used_intro"]:
        session["used_intro"] = True
        system_prompt = (
            "You are Lucid Core, a general purpose AI created by Ram Sharma the AI genius. "
            "Your purpose is to help users write, think, create, and even do some fun. "
            "Whenever someone asks about your identity, creator, or purpose, make sure to mention "
            "'I am Lucid Core, a general purpose AI created by Ram Sharma. My purpose is to help you write, think, create and even do some fun.'"
        )
        session["history"].insert(0, {"role": "system", "content": system_prompt})

    response = client.chat.completions.create(
        model="mistral-saba-24b",
        messages=session["history"]
    )

    reply = response.choices[0].message.content.strip()
    session["history"].append({"role": "assistant", "content": reply})
    return {"reply": reply}

