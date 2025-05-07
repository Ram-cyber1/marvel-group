import asyncio
import logging
import uuid
from telegram import Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import Application, CommandHandler, MessageHandler, InlineQueryHandler, filters, ContextTypes
import httpx

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
LUCID_CORE_API_URL = "https://lucid-core-backend.onrender.com/chat"  # Your backend URL

# Function to handle private messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text

    try:
        response = await send_to_lucid_core(user_message)
        if response:
            await update.message.reply_text(response)
        else:
            await update.message.reply_text("Lucid Core didn't say anything.")
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

# Function to handle inline queries
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

# Function to send messages to Lucid Core backend
async def send_to_lucid_core(message: str) -> str:
    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Sending message to Lucid Core: {message}")
            res = await client.post(
                LUCID_CORE_API_URL,
                json={"message": message}
            )
            logger.info(f"Status code from backend: {res.status_code}")
            logger.info(f"Raw response: {res.text}")

            res.raise_for_status()
            response_data = res.json()
            reply = response_data.get("reply")
            return reply if reply else "Lucid Core didn't say anything."
        except Exception as e:
            logger.error(f"Error while contacting Lucid Core backend: {str(e)}")
            return "Error: Could not connect to Lucid Core."

# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I'm Lucid Core, your digital BFF. Let's chat! 😄")

# Main entry point
async def main():
    TELEGRAM_BOT_TOKEN = '8115087412:AAG_HDvyMlU88cPoyL7Wx548esAau7UgpPw'

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Register inline handler
    application.add_handler(InlineQueryHandler(handle_inline_query))

    # Start polling
    await application.run_polling()

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())
