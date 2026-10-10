FROM python:3.11-slim

# Устанавливаем системные зависимости в один слой
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Настраиваем рабочую директорию и переменные окружения
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Копируем и устанавливаем зависимости отдельным слоем, чтобы он кэшировался.
# CPU-сборка torch: без неё на linux/x86_64 ставится CUDA-версия на несколько ГБ
COPY req.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r req.txt

# Сервис работает не от root
RUN useradd --create-home --uid 1000 app

# Копируем исходный код и веса (лишнее отсекает .dockerignore)
COPY --chown=app:app . .
USER app

# Запуск BentoML сервиса
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3000/readyz', timeout=4)"
CMD ["bentoml", "serve", "service:PlantSegmentator", "--port", "3000"]
