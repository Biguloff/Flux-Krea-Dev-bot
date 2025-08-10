import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Any

import replicate
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    CallbackQueryHandler,
    filters,
)

# === Конфигурация ===
import config

# === БД (SQLAlchemy async) ===
from sqlalchemy import (
    String,
    BigInteger,
    Integer,
    DateTime,
    JSON,
    Text,
    UniqueConstraint,
    select,
    text as sql_text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, Mapped, mapped_column

# Настройки по умолчанию для генерации
DEFAULTS = {
    "guidance": 3.0,
    "output_quality": 95,
    "num_outputs": 1,
    "aspect_ratio": "1:1",
    "output_format": "webp",
    "go_fast": True,
}

# Допустимые значения/диапазоны
ALLOWED_ASPECT_RATIOS = {
    "1:1", "16:9", "21:9", "3:2", "2:3", "4:5", "5:4", "3:4", "4:3", "9:16", "9:21"
}
ALLOWED_OUTPUT_FORMATS = {"webp", "jpg", "png"}
ALLOWED_MEGAPIXELS = {"1", "0.25"}
ALLOWED_KEYS = {
    "seed",
    "image",
    "go_fast",
    "guidance",
    "megapixels",
    "num_outputs",
    "aspect_ratio",
    "output_format",
    "output_quality",
    "prompt_strength",
    "num_inference_steps",
    "disable_safety_checker",
}

# Пресеты для быстрого выбора под площадки
PRESETS: dict[str, dict[str, Any]] = {
    "instagram_square": {"aspect_ratio": "1:1", "output_format": "jpg", "output_quality": 95, "guidance": 2.5, "megapixels": "1"},
    "instagram_story": {"aspect_ratio": "9:16", "output_format": "jpg", "output_quality": 95, "guidance": 2.5, "megapixels": "1"},
    "youtube_thumbnail": {"aspect_ratio": "16:9", "output_format": "jpg", "output_quality": 95, "guidance": 3.0, "megapixels": "1"},
    "tiktok_vertical": {"aspect_ratio": "9:16", "output_format": "jpg", "output_quality": 95, "guidance": 2.5},
    "pinterest_pin": {"aspect_ratio": "2:3", "output_format": "jpg", "output_quality": 95, "guidance": 2.5},
    "x_post_4_5": {"aspect_ratio": "4:5", "output_format": "jpg", "output_quality": 95, "guidance": 2.5},
    "x_post_16_9": {"aspect_ratio": "16:9", "output_format": "jpg", "output_quality": 95, "guidance": 2.5},
    "reels_vertical": {"aspect_ratio": "9:16", "output_format": "jpg", "output_quality": 95, "guidance": 2.5},
}

MODEL_ID = "black-forest-labs/flux-krea-dev"

# Логи
_log_level = getattr(logging, (config.LOG_LEVEL or "INFO").upper(), logging.INFO)
logging.basicConfig(level=_log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === SQLAlchemy: модели и сессии ===
Base = declarative_base()
async_engine = create_async_engine(config.DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionMaker = async_sessionmaker(async_engine, expire_on_commit=False)


class UserPref(Base):
    __tablename__ = "user_prefs"
    __table_args__ = (
        UniqueConstraint("user_id", "key", name="uq_user_prefs_user_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    value_json: Mapped[Any] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class UserAuth(Base):
    __tablename__ = "user_auth"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True, nullable=False)
    is_authenticated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class GenerationLog(Base):
    __tablename__ = "generation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    params: Mapped[Any] = mapped_column(JSON, nullable=True)
    outputs_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="success")
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


async def init_db() -> None:
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def acquire_singleton_lock(lock_key: int = 987654321) -> bool:
    """Пытается получить advisory lock в Postgres. Возвращает True, если замок получен."""
    async with async_engine.connect() as conn:
        result = await conn.execute(sql_text("SELECT pg_try_advisory_lock(:k)"), {"k": lock_key})
        value = result.scalar()
        return bool(value)


async def db_set_pref(user_id: int, key: str, value: Any) -> None:
    now_dt = datetime.now(timezone.utc)
    async with AsyncSessionMaker() as session:
        stmt = pg_insert(UserPref).values(
            user_id=user_id,
            key=key,
            value_json=value,
            updated_at=now_dt,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[UserPref.__table__.c.user_id, UserPref.__table__.c.key],
            set_={"value_json": value, "updated_at": now_dt},
        )
        await session.execute(stmt)
        await session.commit()


async def db_get_prefs(user_id: int) -> dict:
    async with AsyncSessionMaker() as session:
        result = await session.execute(
            select(UserPref.key, UserPref.value_json).where(UserPref.user_id == user_id)
        )
        rows = result.all()
        return {k: v for (k, v) in rows}


async def db_set_auth(user_id: int, is_authed: bool) -> None:
    now_dt = datetime.now(timezone.utc)
    async with AsyncSessionMaker() as session:
        stmt = pg_insert(UserAuth).values(
            user_id=user_id,
            is_authenticated=1 if is_authed else 0,
            updated_at=now_dt,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[UserAuth.__table__.c.user_id],
            set_={"is_authenticated": (1 if is_authed else 0), "updated_at": now_dt},
        )
        await session.execute(stmt)
        await session.commit()


async def db_is_authed(user_id: int) -> bool:
    async with AsyncSessionMaker() as session:
        result = await session.execute(
            select(UserAuth.is_authenticated).where(UserAuth.user_id == user_id)
        )
        row = result.first()
        return bool(row and row[0] == 1)


async def db_log_generation(
    user_id: int,
    prompt: str,
    params: dict,
    outputs_count: int,
    status: str,
    duration_ms: int,
) -> None:
    async with AsyncSessionMaker() as session:
        log = GenerationLog(
            user_id=user_id,
            prompt=prompt,
            params=params,
            outputs_count=outputs_count,
            status=status,
            duration_ms=duration_ms,
        )
        session.add(log)
        await session.commit()


# === Утилиты настроек в контексте ===

def filter_allowed_keys(values: dict) -> dict:
    return {k: v for k, v in values.items() if k in ALLOWED_KEYS}


def coerce_param_types(params: dict) -> dict:
    coerced = {}
    for key, value in params.items():
        k = key.strip()
        v = value
        if k in {"guidance", "prompt_strength"}:
            try:
                coerced[k] = float(v)
            except Exception:
                continue
        elif k in {"num_outputs", "num_inference_steps", "output_quality", "seed"}:
            try:
                coerced[k] = int(v)
            except Exception:
                continue
        elif k in {"go_fast", "disable_safety_checker"}:
            lowered = str(v).strip().lower()
            coerced[k] = lowered in {"true", "1", "yes", "y", "да", "on"}
        elif k in {"aspect_ratio", "output_format", "megapixels", "image"}:
            coerced[k] = str(v)
        else:
            coerced[k] = v
    return coerced


def validate_and_coerce_single(key: str, raw_value: str):
    key = key.strip()
    value = raw_value.strip()
    if key == "guidance":
        val = float(value)
        if not (0 <= val <= 10):
            raise ValueError("guidance должен быть от 0 до 10")
        return val
    if key == "prompt_strength":
        val = float(value)
        if not (0 <= val <= 1):
            raise ValueError("prompt_strength должен быть от 0 до 1")
        return val
    if key == "num_inference_steps":
        val = int(value)
        if not (1 <= val <= 50):
            raise ValueError("num_inference_steps должен быть 1..50")
        return val
    if key == "output_quality":
        val = int(value)
        if not (0 <= val <= 100):
            raise ValueError("output_quality должен быть 0..100")
        return val
    if key == "num_outputs":
        val = int(value)
        if not (1 <= val <= 4):
            raise ValueError("num_outputs должен быть 1..4")
        return val
    if key == "seed":
        return int(value)
    if key == "aspect_ratio":
        if value not in ALLOWED_ASPECT_RATIOS:
            raise ValueError(f"aspect_ratio должен быть одним из: {', '.join(sorted(ALLOWED_ASPECT_RATIOS))}")
        return value
    if key == "output_format":
        if value not in ALLOWED_OUTPUT_FORMATS:
            raise ValueError(f"output_format: {', '.join(sorted(ALLOWED_OUTPUT_FORMATS))}")
        return value
    if key == "megapixels":
        if value not in ALLOWED_MEGAPIXELS:
            raise ValueError(f"megapixels: {', '.join(sorted(ALLOWED_MEGAPIXELS))}")
        return value
    if key == "go_fast" or key == "disable_safety_checker":
        lowered = value.lower()
        return lowered in {"true", "1", "yes", "y", "да", "on"}
    if key == "image":
        return value
    raise ValueError("Недопустимый параметр")


def get_user_prefs(context: ContextTypes.DEFAULT_TYPE) -> dict:
    return context.user_data.get("prefs", {})


def set_user_pref(context: ContextTypes.DEFAULT_TYPE, key: str, value):
    prefs = context.user_data.get("prefs", {})
    prefs[key] = value
    context.user_data["prefs"] = prefs


def prefs_to_text(prefs: dict) -> str:
    if not prefs:
        return "(не заданы)"
    keys = sorted(prefs.keys())
    lines = [f"{k}: {prefs[k]}" for k in keys]
    return "\n".join(lines)


async def ensure_prefs_loaded(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get("prefs_loaded"):
        return
    user = update.effective_user
    if not user:
        return
    db_prefs = await db_get_prefs(user.id)
    if db_prefs:
        # Загружаем в контекст
        context.user_data["prefs"] = db_prefs
    context.user_data["prefs_loaded"] = True


async def ensure_authenticated(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    if not user:
        return False
    # проверяем из кеша
    if context.user_data.get("is_authed") is True:
        return True
    # проверяем в БД
    if await db_is_authed(user.id):
        context.user_data["is_authed"] = True
        return True
    # не авторизован — подсказка
    await (update.effective_message or update.message).reply_text(
        "Требуется авторизация. Введите: /login <пароль>"
    )
    return False


# === Пресеты ===

def preset_human_name(key: str) -> str:
    mapping = {
        "instagram_square": "Instagram (квадрат)",
        "instagram_story": "Instagram (сториз 9:16)",
        "youtube_thumbnail": "YouTube Thumbnail (16:9)",
        "tiktok_vertical": "TikTok (9:16)",
        "pinterest_pin": "Pinterest Pin (2:3)",
        "x_post_4_5": "X/Twitter (4:5)",
        "x_post_16_9": "X/Twitter (16:9)",
        "reels_vertical": "Reels (9:16)",
    }
    return mapping.get(key, key)


async def apply_preset(update: Update, context: ContextTypes.DEFAULT_TYPE, preset_name: str) -> None:
    await ensure_prefs_loaded(update, context)
    preset = PRESETS.get(preset_name)
    if not preset:
        await update.effective_message.reply_text("Неизвестный пресет. Используйте /presets для списка.")
        return
    user = update.effective_user
    filtered = filter_allowed_keys(preset)
    # Сохраняем все пары ключ-значение
    if user:
        for k, v in filtered.items():
            await db_set_pref(user.id, k, v)
            set_user_pref(context, k, v)
    await update.effective_message.reply_text(
        f"Применён пресет: {preset_human_name(preset_name)}\n\n" + prefs_to_text(filtered)
    )


async def cmd_presets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_authenticated(update, context):
        return
    # Список пресетов с кнопками
    rows = []
    row: list[InlineKeyboardButton] = []
    for idx, name in enumerate(PRESETS.keys(), start=1):
        row.append(InlineKeyboardButton(preset_human_name(name), callback_data=f"preset:{name}"))
        if idx % 2 == 0:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    kb = InlineKeyboardMarkup(rows)
    await update.message.reply_text("Выберите пресет:", reply_markup=kb)


async def cmd_preset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_authenticated(update, context):
        return
    args = (update.message.text or "").split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_text("Использование: /preset <имя>. Список: /presets")
        return
    name = args[1].strip()
    await apply_preset(update, context, name)


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_authenticated(update, context):
        return
    q = update.callback_query
    if not q or not q.data:
        return
    if q.data.startswith("preset:"):
        name = q.data.split(":", 1)[1]
        await q.answer()
        # Применяем и отвечаем в том же чате
        await apply_preset(update, context, name)
        return


async def send_typing_animation(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Показывает в чате анимацию генерации (длительный typing + спиннер-сообщение)."""
    # Постоянный чат-экшен typing
    async def typing_action():
        try:
            while True:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            return

    # Спиннер-сообщение
    spinner_frames = [
        "Генерация… ⠋",
        "Генерация… ⠙",
        "Генерация… ⠹",
        "Генерация… ⠸",
        "Генерация… ⠼",
        "Генерация… ⠴",
        "Генерация… ⠦",
        "Генерация… ⠧",
        "Генерация… ⠇",
        "Генерация… ⠏",
    ]

    typing_task = asyncio.create_task(typing_action())
    message = await context.bot.send_message(chat_id=chat_id, text=spinner_frames[0])

    async def spinner_animation():
        i = 0
        try:
            while True:
                i = (i + 1) % len(spinner_frames)
                await message.edit_text(spinner_frames[i])
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            return

    spinner_task = asyncio.create_task(spinner_animation())
    return typing_task, spinner_task, message


async def stop_animation(tasks):
    if not tasks:
        return
    typing_task, spinner_task, message = tasks
    for t in (typing_task, spinner_task):
        if t and not t.done():
            t.cancel()
    # Спрячем спиннер
    try:
        await message.delete()
    except Exception:
        pass


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_prefs_loaded(update, context)
    await update.message.reply_text(
        "Отправьте текстовый промпт (и при желании параметры), и я сгенерирую изображение.\n\n"
        "Также можно отправить фото с подписью (caption) — сделаю img2img.\n\n"
        "Команды настроек: \n"
        "/login <пароль> — авторизация (по умолчанию ss1234, можно изменить в .env).\n"
        "/presets — выбрать готовый пресет под площадку.\n"
        "/preset <имя> — применить пресет по имени (см. /presets).\n"
        "/config — показать текущие настройки.\n"
        "/set <ключ> <значение> — сохранить настройку (например: /set aspect_ratio 16:9).\n"
        "/reset — сбросить сохранённые настройки.\n\n"
        "Примеры промптов:\n"
        "1) a photorealistic portrait of a young woman, soft studio light\n"
        "2) prompt: ultra-detailed cyberpunk city at night; aspect_ratio: 16:9; guidance: 2.5; num_outputs: 2\n\n"
        "Параметры: aspect_ratio, guidance, num_outputs, num_inference_steps, output_quality, output_format, seed, go_fast, disable_safety_checker, megapixels, prompt_strength"
    )


def parse_parameters(text: str) -> tuple[str, dict]:
    """Парсит формат: 'prompt: ...; aspect_ratio: 16:9; num_outputs: 2' или просто текст.
    Возвращает (prompt, params)."""
    if "prompt:" in text:
        parts = [p.strip() for p in text.split(";") if p.strip()]
        prompt_value = ""
        params: dict = {}
        for part in parts:
            if part.lower().startswith("prompt:"):
                prompt_value = part.split(":", 1)[1].strip()
            else:
                if ":" in part:
                    key, value = part.split(":", 1)
                    params[key.strip()] = value.strip()
        if not prompt_value:
            prompt_value = text
        return prompt_value, params
    else:
        return text.strip(), {}


async def cmd_config(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_authenticated(update, context):
        return
    await ensure_prefs_loaded(update, context)
    prefs = get_user_prefs(context)
    effective = dict(DEFAULTS)
    effective.update(filter_allowed_keys(prefs))
    lines = ["Текущие настройки (по умолчанию + ваши):"]
    for k in sorted(effective.keys()):
        lines.append(f"- {k}: {effective[k]}")
    lines.append("\nИзменить: /set <ключ> <значение>\nСброс: /reset")
    await update.message.reply_text("\n".join(lines))


async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_authenticated(update, context):
        return
    await ensure_prefs_loaded(update, context)
    # Поддерживаем 2 формата: /set key value  ИЛИ  /set key: value
    args_text = (update.message.text or "").split(maxsplit=1)
    if len(args_text) < 2:
        await update.message.reply_text("Использование: /set <ключ> <значение>  или  /set ключ: значение")
        return
    rest = args_text[1].strip()
    if ":" in rest:
        key, value = rest.split(":", 1)
        key = key.strip()
        value = value.strip()
    else:
        parts = rest.split(maxsplit=1)
        if len(parts) < 2:
            await update.message.reply_text("Использование: /set <ключ> <значение>")
            return
        key, value = parts[0].strip(), parts[1].strip()

    if key not in ALLOWED_KEYS:
        await update.message.reply_text(
            "Недопустимый ключ. Разрешено: " + ", ".join(sorted(ALLOWED_KEYS))
        )
        return

    try:
        typed_value = validate_and_coerce_single(key, value)
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")
        return

    # Сохраняем в БД и в контекст
    user = update.effective_user
    if user:
        await db_set_pref(user.id, key, typed_value)
    set_user_pref(context, key, typed_value)

    await update.message.reply_text(f"Сохранено: {key} = {typed_value}")


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_authenticated(update, context):
        return
    context.user_data.pop("prefs", None)
    context.user_data.pop("prefs_loaded", None)
    await update.message.reply_text("Сохранённые настройки сброшены (в контексте). Чтобы очистить их в БД, используйте /set для перезаписи или вручную очистите таблицу.")


def replicate_run_sync(inputs: dict):
    # Передаём токен replicate через переменную окружения из config
    if config.REPLICATE_API_TOKEN:
        os.environ["REPLICATE_API_TOKEN"] = config.REPLICATE_API_TOKEN
    return replicate.run(MODEL_ID, input=inputs)


async def generate_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str, user_params: dict, image_url: str | None = None) -> None:
    if not await ensure_authenticated(update, context):
        return
    await ensure_prefs_loaded(update, context)
    chat_id = update.effective_chat.id
    tasks = await send_typing_animation(context, chat_id)
    started = time.perf_counter()
    status = "success"
    try:
        # База: DEFAULTS + сохранённые настройки пользователя + параметры из сообщения
        prefs = filter_allowed_keys(get_user_prefs(context))
        inputs = {**DEFAULTS}
        inputs.update(prefs)
        typed_msg_params = coerce_param_types(user_params)
        inputs.update(filter_allowed_keys(typed_msg_params))

        inputs["prompt"] = prompt
        if image_url:
            inputs["image"] = image_url

        # Запуск синхронной генерации в пуле
        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(None, replicate_run_sync, inputs)

        # Сбор ссылок
        urls: List[str] = []
        for item in output:
            try:
                urls.append(item.url())
            except Exception:
                urls.append(str(item))

        if not urls:
            raise RuntimeError("Replicate не вернул результаты")

        # Отправка фотографий
        if len(urls) == 1:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=urls[0],
                caption=f"Готово!\naspect_ratio: {inputs.get('aspect_ratio')}\nquality: {inputs.get('output_quality')}"
            )
        else:
            media = []
            from telegram import InputMediaPhoto
            for u in urls:
                media.append(InputMediaPhoto(media=u))
            await context.bot.send_media_group(chat_id=chat_id, media=media)

    except Exception as e:
        status = "error"
        logger.exception("Ошибка генерации")
        await context.bot.send_message(chat_id=chat_id, text=f"Произошла ошибка: {e}")
    finally:
        duration_ms = int((time.perf_counter() - started) * 1000)
        try:
            user = update.effective_user
            if user:
                # Логируем параметры без бинарных данных
                params_for_log = {k: v for k, v in inputs.items() if k not in {"prompt", "image"}}
                await db_log_generation(
                    user_id=user.id,
                    prompt=prompt,
                    params=params_for_log,
                    outputs_count=len(urls) if 'urls' in locals() else 0,
                    status=status,
                    duration_ms=duration_ms,
                )
        except Exception:
            logger.exception("Не удалось записать лог генерации")
        await stop_animation(tasks)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_authenticated(update, context):
        return
    text = update.message.text or ""
    prompt, params = parse_parameters(text)
    if not prompt:
        await update.message.reply_text("Введите описание изображения (prompt)")
        return
    await generate_and_send(update, context, prompt, params)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_authenticated(update, context):
        return
    if not update.message or not update.message.photo:
        return
    # Берем самое большое превью
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_url = file.file_path  # URL Telegram, доступный по токену

    caption = update.message.caption or ""
    prompt, params = parse_parameters(caption) if caption else ("Image-to-Image", {})

    await generate_and_send(update, context, prompt, params, image_url=file_url)


async def cmd_login(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    args = (update.message.text or "").split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_text("Использование: /login <пароль>")
        return
    password = args[1].strip()
    if password == (config.AUTH_PASSWORD or ""):
        if user:
            await db_set_auth(user.id, True)
        context.user_data["is_authed"] = True
        await update.message.reply_text("Авторизация успешна. Можете продолжать.")
    else:
        await update.message.reply_text("Неверный пароль. Попробуйте ещё раз.")


def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN в .env или окружении")
    if not config.REPLICATE_API_TOKEN:
        raise RuntimeError("Не задан REPLICATE_API_TOKEN в .env или окружении")

    # Инициализация БД (создание таблиц) и попытка стать единственным экземпляром
    async def _bootstrap() -> bool:
        await init_db()
        return await acquire_singleton_lock()

    got_lock = asyncio.run(_bootstrap())
    if not got_lock:
        logger.error(
            "Другой экземпляр уже запущен (advisory lock не получен). Завершение работы, чтобы предотвратить 409 Conflict."
        )
        return

    # В Python 3.13 get_event_loop() может падать, если цикл не создан.
    # Создадим новый event loop для run_polling, если текущего нет.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("login", cmd_login))
    app.add_handler(CommandHandler("presets", cmd_presets))
    app.add_handler(CommandHandler("preset", cmd_preset))
    app.add_handler(CommandHandler("config", cmd_config))
    app.add_handler(CommandHandler("settings", cmd_config))
    app.add_handler(CommandHandler("set", cmd_set))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling()


if __name__ == "__main__":
    main()