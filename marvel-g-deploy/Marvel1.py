#!/usr/bin/env python3
"""
🦸‍♂️ Marvel Group Manager Bot with Health Check
Created by Ram Sharma | @Lucid_Core
"""

import logging
import json
import asyncio
import threading
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, jsonify
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, ChatPermissions
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot token (keep secret in production!)
BOT_TOKEN = "7720248790:AAGiKq9pT0MdGABJiwK5hkds9K4S0f9tkGU"

# Flask app for health check
flask_app = Flask(__name__)

# File paths for data storage
LOCKS_FILE = "group_locks.json"
WARNINGS_FILE = "warnings.json"
WELCOME_FILE = "welcome_messages.json"

# Load or initialize lock settings
try:
    with open(LOCKS_FILE, "r") as f:
        group_locks = json.load(f)
except FileNotFoundError:
    group_locks = {}

def save_locks():
    with open(LOCKS_FILE, "w") as f:
        json.dump(group_locks, f)

# Load or initialize warnings
try:
    with open(WARNINGS_FILE, "r") as f:
        warnings_data = json.load(f)
except FileNotFoundError:
    warnings_data = {}

def save_warnings():
    with open(WARNINGS_FILE, "w") as f:
        json.dump(warnings_data, f)

# Load or initialize welcome messages
try:
    with open(WELCOME_FILE, "r") as f:
        welcome_messages = json.load(f)
except FileNotFoundError:
    welcome_messages = {}

def save_welcome_messages():
    with open(WELCOME_FILE, "w") as f:
        json.dump(welcome_messages, f)

# Health check endpoint
@flask_app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'bot_status': 'running',
        'uptime': 'active'
    })

@flask_app.route('/')
def home():
    return jsonify({
        'bot': 'Marvel Group Manager Bot',
        'status': 'online',
        'endpoints': {
            'health': '/health',
            'ping': '/ping'
        }
    })

@flask_app.route('/ping')
def ping():
    return jsonify({'message': 'pong', 'timestamp': datetime.utcnow().isoformat()})

# Helper: Check if user is admin
async def is_user_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat = update.effective_chat
    
    try:
        member = await chat.get_member(user_id)
        return member.status in ("administrator", "creator")
    except Exception:
        return False

# Decorator for admin-only commands
def admin_only(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await is_user_admin(update, context):
            await update.message.reply_text("❌ You must be an admin to use this command.")
            return
        return await func(update, context)
    return wrapper

# Start command with enhanced buttons
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("ℹ️ Help", callback_data="help"),
         InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
        [InlineKeyboardButton("📜 Rules", callback_data="rules"),
         InlineKeyboardButton("🛡️ Admin Panel", callback_data="admin_panel")],
        [InlineKeyboardButton("📊 Stats", callback_data="stats")]
    ])
    
    welcome_text = (
        "🦸‍♂️ *Marvel Group Manager Bot*\n\n"
        "Welcome to the most powerful Telegram group management bot!\n\n"
        "🛡️ *Features:*\n"
        "• Advanced moderation tools\n"
        "• User warnings system\n"
        "• Content locks (media, links, stickers)\n"
        "• Welcome/goodbye messages\n"
        "• Report system for admins\n\n"
        "Click the buttons below to explore!"
    )
    
    await update.message.reply_text(
        welcome_text,
        reply_markup=keyboard,
        parse_mode="Markdown"
    )

# Enhanced callback handler with more options
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "help":
        help_text = (
            "🛠️ *Help Menu*\n\n"
            "*Admin Commands:*\n"
            "/ban - Ban a user (reply to message)\n"
            "/kick - Kick a user (reply to message)\n"
            "/mute - Mute a user temporarily\n"
            "/unmute - Unmute a user\n"
            "/warn - Warn a user (3 warns = ban)\n"
            "/unwarn - Remove a warning\n"
            "/lock - Lock content types\n"
            "/unlock - Unlock content types\n"
            "/setwelcome - Set custom welcome message\n"
            "/welcome - Toggle welcome messages on/off\n\n"
            "*User Commands:*\n"
            "/start - Show welcome message\n"
            "/warns - Check warning count\n"
            "/report - Report a message to admins\n\n"
            "*Lock Types:* media, links, stickers"
        )
        back_button = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔙 Back", callback_data="main_menu")
        ]])
        await query.edit_message_text(help_text, reply_markup=back_button, parse_mode="Markdown")

    elif query.data == "settings":
        settings_text = (
            "⚙️ *Settings Panel*\n\n"
            "🔒 *Available Locks:*\n"
            "• Media Lock - Prevents photos/videos\n"
            "• Link Lock - Blocks all links\n"
            "• Sticker Lock - Removes stickers\n\n"
            "⚠️ *Warning System:*\n"
            "• 3 strikes policy\n"
            "• Auto-ban after 3rd warning\n"
            "• Admins can warn/unwarn users\n\n"
            "Use /lock and /unlock commands to manage content."
        )
        back_button = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔙 Back", callback_data="main_menu")
        ]])
        await query.edit_message_text(settings_text, reply_markup=back_button, parse_mode="Markdown")

    elif query.data == "rules":
        rules_text = (
            "📜 *Group Rules*\n\n"
            "1️⃣ Be respectful to all members\n"
            "2️⃣ No spam or excessive promotion\n"
            "3️⃣ Follow admin instructions promptly\n"
            "4️⃣ No offensive language or harassment\n"
            "5️⃣ Stay on topic and keep discussions civil\n"
            "6️⃣ No sharing of inappropriate content\n\n"
            "⚠️ *Violations may result in warnings or bans*"
        )
        back_button = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔙 Back", callback_data="main_menu")
        ]])
        await query.edit_message_text(rules_text, reply_markup=back_button, parse_mode="Markdown")

    elif query.data == "admin_panel":
        if not await is_user_admin_callback(query, context):
            await query.edit_message_text("❌ Admin access required!")
            return
            
        admin_text = (
            "🛡️ *Admin Control Panel*\n\n"
            "*Quick Actions:*\n"
            "• Reply to messages with /ban, /kick, /mute\n"
            "• Use /warn system for progressive discipline\n"
            "• Lock content types with /lock command\n\n"
            "*Current Status:*\n"
            f"• Active Warnings: {sum(len(users) for users in warnings_data.values())}\n"
            f"• Groups with Locks: {len(group_locks)}\n\n"
            "Use commands in chat for full functionality."
        )
        back_button = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔙 Back", callback_data="main_menu")
        ]])
        await query.edit_message_text(admin_text, reply_markup=back_button, parse_mode="Markdown")

    elif query.data == "stats":
        chat_id = str(query.message.chat.id)
        warn_count = len(warnings_data.get(chat_id, {}))
        lock_count = len(group_locks.get(chat_id, {}))
        
        stats_text = (
            "📊 *Group Statistics*\n\n"
            f"👥 *Chat:* {query.message.chat.title or 'Private Chat'}\n"
            f"⚠️ *Users with Warnings:* {warn_count}\n"
            f"🔒 *Active Locks:* {lock_count}\n"
            f"🤖 *Bot Status:* Online & Active\n\n"
            "Statistics update in real-time!"
        )
        back_button = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔙 Back", callback_data="main_menu")
        ]])
        await query.edit_message_text(stats_text, reply_markup=back_button, parse_mode="Markdown")

    elif query.data == "main_menu":
        # Return to main menu
        await start_command_callback(query, context)

# Helper for callback admin check
async def is_user_admin_callback(query, context):
    user_id = query.from_user.id
    chat = query.message.chat
    try:
        member = await chat.get_member(user_id)
        return member.status in ("administrator", "creator")
    except Exception:
        return False

# Callback version of start command
async def start_command_callback(query, context):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("ℹ️ Help", callback_data="help"),
         InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
        [InlineKeyboardButton("📜 Rules", callback_data="rules"),
         InlineKeyboardButton("🛡️ Admin Panel", callback_data="admin_panel")],
        [InlineKeyboardButton("📊 Stats", callback_data="stats")]
    ])
    
    welcome_text = (
        "🦸‍♂️ *Marvel Group Manager Bot*\n\n"
        "Welcome to the most powerful Telegram group management bot!\n\n"
        "🛡️ *Features:*\n"
        "• Advanced moderation tools\n"
        "• User warnings system\n"
        "• Content locks (media, links, stickers)\n"
        "• Welcome/goodbye messages\n"
        "• Report system for admins\n\n"
        "Click the buttons below to explore!"
    )
    
    await query.edit_message_text(welcome_text, reply_markup=keyboard, parse_mode="Markdown")

# Ban command
@admin_only
async def ban_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Reply to the user you want to ban.")
        return

    user_to_ban = update.message.reply_to_message.from_user
    try:
        await context.bot.ban_chat_member(update.effective_chat.id, user_to_ban.id)
        await update.message.reply_text(f"🚫 {user_to_ban.first_name} has been banned!")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Failed to ban: {str(e)}")

# Kick command
@admin_only
async def kick_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Reply to the user you want to kick.")
        return

    user_to_kick = update.message.reply_to_message.from_user
    try:
        await context.bot.ban_chat_member(update.effective_chat.id, user_to_kick.id)
        await context.bot.unban_chat_member(update.effective_chat.id, user_to_kick.id)
        await update.message.reply_text(f"👢 {user_to_kick.first_name} has been kicked!")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Failed to kick: {str(e)}")

# Enhanced mute command with duration
@admin_only
async def mute_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Reply to the user you want to mute.")
        return

    user = update.message.reply_to_message.from_user
    duration_arg = context.args[0] if context.args else "1h"

    # Duration parsing
    try:
        time_unit = duration_arg[-1].lower()
        time_val = int(duration_arg[:-1])
        if time_unit == 'm':
            duration = timedelta(minutes=time_val)
        elif time_unit == 'h':
            duration = timedelta(hours=time_val)
        elif time_unit == 'd':
            duration = timedelta(days=time_val)
        else:
            raise ValueError
    except (ValueError, IndexError):
        await update.message.reply_text("⚠️ Invalid time format. Use: 5m, 1h, 1d")
        return

    try:
        until = datetime.utcnow() + duration
        await context.bot.restrict_chat_member(
            update.effective_chat.id,
            user.id,
            permissions=ChatPermissions(can_send_messages=False),
            until_date=until
        )
        await update.message.reply_text(f"🔇 {user.first_name} has been muted for {duration_arg}!")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Failed to mute: {str(e)}")

# Unmute command
@admin_only
async def unmute_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Reply to the user you want to unmute.")
        return

    user_id = update.message.reply_to_message.from_user.id
    try:
        await context.bot.restrict_chat_member(
            chat_id=update.effective_chat.id,
            user_id=user_id,
            permissions=ChatPermissions(
                can_send_messages=True,
                can_send_media_messages=True,
                can_send_other_messages=True,
                can_add_web_page_previews=True,
            )
        )
        await update.message.reply_text("🔊 User has been unmuted!")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Failed to unmute: {str(e)}")

# Lock command
@admin_only
async def lock_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    args = context.args

    if len(args) != 1 or args[0] not in ['media', 'links', 'stickers']:
        await update.message.reply_text("Usage: /lock <media|links|stickers>")
        return

    lock_type = args[0]
    group_locks.setdefault(chat_id, {})[lock_type] = True
    save_locks()
    await update.message.reply_text(f"🔒 {lock_type.capitalize()} locked!")

# Unlock command
@admin_only
async def unlock_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    args = context.args

    if len(args) != 1 or args[0] not in ['media', 'links', 'stickers']:
        await update.message.reply_text("Usage: /unlock <media|links|stickers>")
        return

    lock_type = args[0]
    if chat_id in group_locks and lock_type in group_locks[chat_id]:
        group_locks[chat_id][lock_type] = False
        save_locks()
        await update.message.reply_text(f"🔓 {lock_type.capitalize()} unlocked!")
    else:
        await update.message.reply_text(f"🔓 {lock_type.capitalize()} was already unlocked.")

# Lock filter to enforce content restrictions
async def lock_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    chat_id = str(msg.chat.id)
    locks = group_locks.get(chat_id, {})

    # Check if user is admin (admins bypass locks)
    if await is_user_admin(update, context):
        return

    # Delete media
    if locks.get("media") and (msg.photo or msg.video or msg.document or msg.audio):
        await msg.delete()
        return

    # Delete links
    if locks.get("links") and msg.entities:
        for entity in msg.entities:
            if entity.type in ['url', 'text_link']:
                await msg.delete()
                return

    # Delete stickers
    if locks.get("stickers") and msg.sticker:
        await msg.delete()
        return

# Warn command
@admin_only
async def warn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Reply to a user's message to warn them.")
        return

    user = update.message.reply_to_message.from_user
    chat_id = str(update.effective_chat.id)
    user_id = str(user.id)

    # Initialize
    warnings_data.setdefault(chat_id, {})
    warnings_data[chat_id].setdefault(user_id, 0)

    warnings_data[chat_id][user_id] += 1
    count = warnings_data[chat_id][user_id]
    save_warnings()

    if count >= 3:
        try:
            await context.bot.ban_chat_member(chat_id, user.id)
            await update.message.reply_text(f"🚫 {user.first_name} has been banned after 3 warnings!")
            warnings_data[chat_id][user_id] = 0  # reset warnings after ban
            save_warnings()
        except Exception as e:
            await update.message.reply_text(f"❌ Failed to ban user: {str(e)}")
    else:
        await update.message.reply_text(f"⚠️ {user.first_name} has been warned. ({count}/3)")

# Unwarn command
@admin_only
async def unwarn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Reply to a user's message to unwarn them.")
        return

    user = update.message.reply_to_message.from_user
    chat_id = str(update.effective_chat.id)
    user_id = str(user.id)

    if warnings_data.get(chat_id, {}).get(user_id, 0) > 0:
        warnings_data[chat_id][user_id] -= 1
        save_warnings()
        new_count = warnings_data[chat_id][user_id]
        await update.message.reply_text(f"✅ {user.first_name}'s warning removed. ({new_count}/3)")
    else:
        await update.message.reply_text(f"ℹ️ {user.first_name} has no warnings to remove.")

# Check warnings command
async def warns_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await update.message.reply_text("ℹ️ Reply to a user's message to check their warnings.")
        return

    user = update.message.reply_to_message.from_user
    chat_id = str(update.effective_chat.id)
    user_id = str(user.id)

    count = warnings_data.get(chat_id, {}).get(user_id, 0)
    await update.message.reply_text(f"👮 {user.first_name} has {count}/3 warnings.")

# Report command
async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ You need to reply to a message to report it.")
        return

    reported_msg = update.message.reply_to_message
    reporter = update.effective_user.first_name
    offender = reported_msg.from_user.first_name
    chat_title = update.effective_chat.title or "this chat"

    report_text = (
        f"🚨 *New Report* in {chat_title}\n"
        f"👤 Reported User: {offender}\n"
        f"🗣 Reported By: {reporter}\n"
        f"📝 Message: {reported_msg.text or 'Media/Special content'}"
    )

    # Send to group admins
    try:
        admins = await context.bot.get_chat_administrators(update.effective_chat.id)
        admin_count = 0
        for admin in admins:
            if not admin.user.is_bot:
                try:
                    await context.bot.send_message(admin.user.id, report_text, parse_mode="Markdown")
                    admin_count += 1
                except Exception:
                    pass  # Admin has blocked the bot or doesn't allow messages

        await update.message.reply_text(f"✅ Report sent to {admin_count} admin(s). Thank you!")
    except Exception as e:
        await update.message.reply_text("⚠️ Failed to send report to admins.")

# Set welcome message command
@admin_only
async def setwelcome_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    
    if not context.args:
        current_welcome = welcome_messages.get(chat_id, {}).get('message', 'No custom welcome set')
        await update.message.reply_text(
            f"📝 *Current Welcome Message:*\n\n{current_welcome}\n\n"
            "*Usage:* `/setwelcome <your message>`\n\n"
            "*Variables you can use:*\n"
            "`{name}` - User's first name\n"
            "`{username}` - User's username\n"
            "`{chat}` - Chat name\n"
            "`{count}` - Member count\n\n"
            "*Example:* `/setwelcome Welcome {name} to {chat}! 🎉`",
            parse_mode="Markdown"
        )
        return
    
    welcome_text = ' '.join(context.args)
    
    # Initialize welcome data for this chat
    welcome_messages.setdefault(chat_id, {})
    welcome_messages[chat_id]['message'] = welcome_text
    welcome_messages[chat_id]['enabled'] = True
    save_welcome_messages()
    
    # Show preview
    preview = welcome_text.format(
        name="John",
        username="@john",
        chat=update.effective_chat.title or "This Chat",
        count="123"
    )
    
    await update.message.reply_text(
        f"✅ *Welcome message updated!*\n\n"
        f"*Preview:*\n{preview}\n\n"
        f"*Original:*\n{welcome_text}",
        parse_mode="Markdown"
    )

# Toggle welcome messages on/off
@admin_only
async def welcome_toggle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    
    # Initialize if doesn't exist
    welcome_messages.setdefault(chat_id, {'enabled': True, 'message': None})
    
    # Toggle the setting
    current_status = welcome_messages[chat_id].get('enabled', True)
    welcome_messages[chat_id]['enabled'] = not current_status
    save_welcome_messages()
    
    status = "enabled" if welcome_messages[chat_id]['enabled'] else "disabled"
    emoji = "✅" if welcome_messages[chat_id]['enabled'] else "❌"
    
    await update.message.reply_text(f"{emoji} Welcome messages are now *{status}*!", parse_mode="Markdown")

# Reset welcome to default
@admin_only
async def resetwelcome_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    
    if chat_id in welcome_messages:
        welcome_messages[chat_id]['message'] = None
        welcome_messages[chat_id]['enabled'] = True
        save_welcome_messages()
        await update.message.reply_text("🔄 Welcome message reset to default!")
    else:
        await update.message.reply_text("ℹ️ No custom welcome message was set.")

# Enhanced welcome handler with custom messages
async def welcome_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    welcome_config = welcome_messages.get(chat_id, {})
    
    # Check if welcome messages are enabled
    if not welcome_config.get('enabled', True):
        return
    
    for member in update.message.new_chat_members:
        if member.is_bot:
            continue
            
        # Get member count
        try:
            member_count = await update.effective_chat.get_member_count()
        except:
            member_count = "many"
        
        # Use custom welcome message if set
        custom_message = welcome_config.get('message')
        if custom_message:
            try:
                welcome_text = custom_message.format(
                    name=member.first_name,
                    username=f"@{member.username}" if member.username else member.first_name,
                    chat=update.effective_chat.title or "this chat",
                    count=member_count
                )
            except KeyError as e:
                # If there's an error in formatting, use default message
                welcome_text = f"👋 Welcome {member.first_name}! There was an error in the welcome message format."
        else:
            # Default welcome message
            welcome_text = (
                f"👋 Welcome to {update.effective_chat.title or 'the group'}, {member.first_name}!\n\n"
                f"🦸‍♂️ I'm Marvel Bot, your group guardian.\n"
                f"👥 You are member #{member_count}!\n\n"
                f"Feel free to explore and make yourself at home!"
            )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📜 Rules", callback_data="rules"),
             InlineKeyboardButton("ℹ️ Help", callback_data="help")]
        ])
        
        await update.message.reply_text(welcome_text, reply_markup=keyboard)

# Goodbye to users who leave
async def goodbye_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.left_chat_member:
        name = update.message.left_chat_member.first_name
        await update.message.reply_text(f"👋 {name} just left the chat. Goodbye!")

def run_flask():
    """Run Flask app in a separate thread"""
    import os
    port = int(os.environ.get('PORT', 5000))
    flask_app.run(host='0.0.0.0', port=port)

# Main function
def main():
    """Main function to run the bot"""
    try:
        # Start Flask server in a separate thread
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        logger.info("Flask health check server started")
        
        # Build the application
        app = ApplicationBuilder().token(BOT_TOKEN).build()

        # Command handlers
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CallbackQueryHandler(button_handler))
        app.add_handler(CommandHandler("ban", ban_command))
        app.add_handler(CommandHandler("kick", kick_command))
        app.add_handler(CommandHandler("mute", mute_command))
        app.add_handler(CommandHandler("unmute", unmute_command))
        app.add_handler(CommandHandler("lock", lock_command))
        app.add_handler(CommandHandler("unlock", unlock_command))
        app.add_handler(CommandHandler("warn", warn_command))
        app.add_handler(CommandHandler("unwarn", unwarn_command))
        app.add_handler(CommandHandler("warns", warns_command))
        app.add_handler(CommandHandler("report", report_command))
        app.add_handler(CommandHandler("setwelcome", setwelcome_command))
        app.add_handler(CommandHandler("welcome", welcome_toggle_command))
        app.add_handler(CommandHandler("resetwelcome", resetwelcome_command))
        
        # Message handlers
        app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, welcome_handler))
        app.add_handler(MessageHandler(filters.StatusUpdate.LEFT_CHAT_MEMBER, goodbye_handler))
        app.add_handler(MessageHandler(filters.ALL & (~filters.COMMAND), lock_filter))

        logger.info("🦸‍♂️ Marvel Group Manager Bot is starting up...")
        logger.info("Bot is now running! Press Ctrl+C to stop.")
        
        # Run the bot
        app.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise

if __name__ == "__main__":
    main()