## Телеграм-бот генерации изображений (FLUX.1 Krea [dev] / Replicate)

Минималистичный бот для генерации изображений через модель FLUX.1 Krea [dev] от Black Forest Labs (Krea) на Replicate. С анимацией процесса, сохранением настроек пользователя (PostgreSQL), пресетами под площадки и Docker-окружением.

Ссылка на модель: [FLUX.1 Krea dev (Replicate)](https://replicate.com/black-forest-labs/flux-krea-dev)


### 1) Требования
- Python 3.11+ (рекомендуется 3.13)
- Git
- Docker и Docker Compose (для контейнерного запуска)
- Аккаунт в Telegram (BotFather токен)
- Токен Replicate API


### 2) Подготовка токенов
- Получите токен бота у BotFather.
- Получите Replicate API Token.


### 3) Клонирование и конфиг
```bash
git clone git@github.com:Biguloff/Flux-Krea-Dev-bot.git
cd Flux-Krea-Dev-bot
```
Создайте файл `.env` (не коммитится):
```bash
cat > .env << 'EOF'
TELEGRAM_BOT_TOKEN=ваш_токен_бота
REPLICATE_API_TOKEN=ваш_replicate_api_token
# Для локального запуска без Docker (пример ниже):
DATABASE_URL=postgresql+asyncpg://localhost/tg_bot
LOG_LEVEL=INFO
EOF
```


### 4) Локальный запуск (без Docker)
- Установите зависимости в виртуальное окружение:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
- Поднимите PostgreSQL (пример для macOS/Homebrew):
```bash
brew services start postgresql
createdb tg_bot || true
```
- Запустите бота:
```bash
python bot.py
```


### 5) Запуск в Docker (рекомендуется)
- Убедитесь, что `.env` создан в корне.
- Поднимите сервисы:
```bash
docker compose up -d --build
```
- Логи бота:
```bash
docker compose logs -f bot
```
Примечания:
- Внутри Docker бот ходит к БД по `postgresql+asyncpg://tg:tg@db:5432/tg_bot` (см. `docker-compose.yml`).
- Порт Postgres наружу не пробрасывается, чтобы не конфликтовать с локальным 5432.


### 6) Использование бота
- Команды:
  - `/start` — краткая справка.
  - `/config` — показать текущие настройки (дефолт + сохранённые).
  - `/set <ключ> <значение>` — сохранить настройку (в БД и сессии), пример: `/set aspect_ratio 16:9`.
  - `/reset` — сбросить сохранённые настройки из сессии.
  - `/presets` — список пресетов (кнопки) под площадки.
  - `/preset <имя>` — применить пресет по имени (см. `/presets`).
- Генерация:
  - Отправьте текст — начнётся генерация (идёт анимация typing + спиннер).
  - Отправьте фото с подписью — режим img2img.
  - Можно задать параметры через `;`:
    ```
    prompt: cyberpunk city at night; aspect_ratio: 16:9; guidance: 2.5; num_outputs: 2; output_quality: 95
    ```

Поддерживаемые параметры (Replicate): `aspect_ratio`, `guidance`, `num_outputs`, `num_inference_steps`, `output_quality`, `output_format`, `seed`, `go_fast`, `disable_safety_checker`, `megapixels`, `prompt_strength`, `image` (для img2img).


### 7) Пресеты (имена)
- `instagram_square`, `instagram_story`, `youtube_thumbnail`, `tiktok_vertical`,
  `pinterest_pin`, `x_post_4_5`, `x_post_16_9`, `reels_vertical`.


### 8) Коммиты и GitHub
- Инициализация (если репо ещё не инициализировано):
```bash
git init
git checkout -B main
git add -A
git commit -m "Initial commit"
```
- Привязать удалённый и запушить:
  - SSH:
    ```bash
    git remote add origin git@github.com:<username>/<repo>.git
    git push -u origin main
    ```
  - HTTPS с PAT:
    ```bash
    git remote add origin https://<username>:<PAT>@github.com/<username>/<repo>.git
    git push -u origin main
    git remote set-url origin https://github.com/<username>/<repo>.git
    ```
- Убедитесь, что у токена есть права (classic: `repo`; fine-grained: доступ к репозиторию и `Contents: Read and write`).


### 9) Деплой на сервер
- Вариант Docker:
```bash
git clone git@github.com:<username>/<repo>.git
cd <repo>
# создайте .env и заполните токены
docker compose up -d --build
```
- Если бот и БД на одном сервере без Docker:
  - используйте `DATABASE_URL=postgresql+asyncpg://USER:PASS@localhost:5432/tg_bot`.
- Если бот в Docker, а БД на хосте:
  - macOS/Windows: `host.docker.internal`
  - Linux: добавьте в compose
    ```yaml
    extra_hosts:
      - "host.docker.internal:host-gateway"
    ```
    затем `DATABASE_URL=postgresql+asyncpg://USER:PASS@host.docker.internal:5432/tg_bot`


### 10) Траблшутинг
- 409 Conflict в Telegram: запущено несколько экземпляров — оставьте один (в Docker — один сервис `bot`).
- Конфликт порта 5432: не пробрасывайте Postgres наружу или смените порт.
- Python 3.13 event loop: в коде добавлено явное создание event loop перед `run_polling()`.
- Нет изображений: проверьте валидность токена Replicate, лимиты и параметры.


### 11) Безопасность
- Не коммитьте `.env` — он игнорируется в `.gitignore`.
- Храните токены в секретах (CI/CD, менеджеры секретов).
- Для БД используйте отдельного пользователя и сложный пароль, закройте внешний доступ если не нужен.


### 12) Структура проекта
```
.
├─ bot.py                 # основной код бота
├─ config.py              # загрузка конфигурации из .env (python-dotenv)
├─ requirements.txt       # зависимости Python
├─ docker-compose.yml     # бот + Postgres
├─ Dockerfile             # образ приложения
├─ .gitignore             # игнор для Git
└─ README.md              # этот файл
```
