import logging
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from chat import get_response
import os
from dotenv import load_dotenv
load_dotenv()

# logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(rf"Halo {user.mention_html()}! Trasbot disini, kamu bisa menanyakan kepadaku apapun tentang Traspac!",reply_markup=ForceReply(selective=True))

async def response(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text_response = await get_response(update.message.text)
    await update.message.reply_text(text_response)


def main() -> None:
    application = Application.builder().token(os.getenv('TELEGRAM_API_KEY')).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, response))

    application.run_polling()


if __name__ == "__main__":
    main()