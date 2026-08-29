# syntax=docker/dockerfile:1
#
# Двофазний ланцюг постачання (п.9).
#
#   Фаза 1 (builder) — єдиний момент, коли є мережа: тягнемо ваги за
#                      models/manifest.lock і звіряємо sha256.
#   Фаза 2 (runtime) — мережі немає й не має бути. Ваги вже всередині,
#                      жодного мережевого клієнта і жодних токенів у шарі.
#
# Перевірка властивості, заради якої все це робиться:
#     docker build -t vsearch .
#     docker run --network=none vsearch pytest -m offline

ARG PYTHON_VERSION=3.12

# ─────────────────────────────── ФАЗА 1: BUILDER ───────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

# Опційні ваги (so400m, large, whisper) роздувають образ у рази,
# тож типово не тягнемо. Для профілю quality: --build-arg INCLUDE_OPTIONAL=1
ARG INCLUDE_OPTIONAL=0

WORKDIR /build
RUN pip install --no-cache-dir "huggingface_hub>=0.24"

COPY models/manifest.lock /build/models/manifest.lock
COPY scripts/fetch_models.py /build/scripts/
COPY src/vsearch/backends/registry.py /build/src/vsearch/backends/registry.py
COPY src/vsearch/backends/__init__.py /build/src/vsearch/backends/__init__.py
COPY src/vsearch/__init__.py /build/src/vsearch/__init__.py

# Ваги качаються ТУТ і тільки тут. Розбіжність sha256 валить збірку —
# саме тому перевірка стоїть на шляху, який неможливо обійти.
RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN_FILE=/run/secrets/hf_token \
    python scripts/fetch_models.py \
        $([ "$INCLUDE_OPTIONAL" = "1" ] && echo --include-optional) \
    && python scripts/fetch_models.py --check

# ─────────────────────────────── ФАЗА 2: RUNTIME ───────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

# Мовні пакети Tesseract ставляться як пакети ОС, а не тягнуться при першому
# виклику: іврит (RTL) і кирилиця потрібні для OCR у контурі.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        tesseract-ocr-heb \
        tesseract-ocr-ukr \
        tesseract-ocr-rus \
        tesseract-ocr-deu \
        tesseract-ocr-pol \
    && rm -rf /var/lib/apt/lists/*

# Ці змінні — другий рубіж після netguard: навіть якщо в коді десь лишиться
# виклик у мережу, бібліотеки HuggingFace його не виконають.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TORCH_HOME=/opt/vsearch/models/torch \
    VSEARCH_MODELS_DIR=/opt/vsearch/models \
    PYTHONPATH=/opt/vsearch/src:/opt/vsearch \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/vsearch

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY eval/ ./eval/
COPY features/ ./features/
COPY tests/ ./tests/
COPY scripts/verify_offline.py scripts/check_licenses.py scripts/benchmark.py ./scripts/

RUN pip install --no-cache-dir ".[test,ml,index,video,ocr,llm,api]"

# Ваги переїжджають із builder-шару. scripts/fetch_models.py сюди СВІДОМО
# не копіюється: у цьому образі його нема чим і нема навіщо запускати.
COPY --from=builder /build/models /opt/vsearch/models

# Доводимо офлайн-контур на етапі збірки, а не на демонстрації в замовника.
RUN python scripts/verify_offline.py && python scripts/check_licenses.py

EXPOSE 8000
CMD ["python", "-m", "vsearch.cli", "serve", "--host", "0.0.0.0"]
