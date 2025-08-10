import os
from dotenv import load_dotenv

# Загружаем .env, если он есть
load_dotenv()

# Токены/ключи
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
REPLICATE_API_TOKEN: str = os.getenv("REPLICATE_API_TOKEN", "")

# База данных
DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://localhost/tg_bot")

# Логи
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")