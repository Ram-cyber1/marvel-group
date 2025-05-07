import asyncio
import logging
import uuid
import os
from fastapi import FastAPI, Request
from telegram import Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import Application, CommandHandler, MessageHandler, InlineQueryHandler, filters, ContextTypes
import httpx

# Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
LUCID_CORE_API_URL = "https://lucid-core-backend.onrender.com/chat"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8115087412:AAG_HDvyMlU88cPoyL7Wx548esAau7UgpPw")
WEBHOOK_URL = "https://telegram-bot-deploy-2.onrender.com/webhook"  # Replace this with your actual webhook URL

# FastAPI app
app = FastAPI()

@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping(request: Request):
    return {"status": "Lucid Core Telegram bot is alive!"}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_bot())

# Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I'm Lucid Core, your digital BFF. Let's chat! 😄")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        response = await send_to_lucid_core(update.message.text)
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

async def handle_inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query
    if not query:
        return
    try:
        response = await send_to_lucid_core(query)
        results = [
            InlineQueryResultArticle(
                id=str(uuid.uuid4()),
                title="Lucid Core's Reply",
                input_message_content=InputTextMessageContent(response),
                description=response[:50] + '...' if len(response) > 50 else response
            )
        ]
        await update.inline_query.answer(results, cache_time=0)
    except Exception as e:
        logger.error(f"Inline query failed: {str(e)}")

async def send_to_lucid_core(message: str) -> str:
    async with httpx.AsyncClient() as client:
        res = await client.post(LUCID_CORE_API_URL, json={"message": message})
        res.raise_for_status()
        return res.json().get("reply", "Lucid Core didn't say anything.")

# Run bot with Webhook
async def run_bot():
    app_bot = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app_bot.add_handler(InlineQueryHandler(handle_inline_query))

    # Set webhook with Telegram API
    await app_bot.bot.set_webhook(WEBHOOK_URL)

    # Start FastAPI app
    await app_bot.initialize()
    logger.info("Webhook set successfully!")
    
# FastAPI endpoint to receive updates
@app.post("/webhook")
async def webhook(update: dict):
    update = Update.de_json(update, Application.builder().token(TELEGRAM_BOT_TOKEN).build().bot)
    await app_bot.process_update(update)
    return {"status": "ok"}


