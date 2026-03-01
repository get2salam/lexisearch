# ============================================================
# LexiSearch — multi-stage Docker image
# ============================================================
# Stage 1: builder — install deps into a venv
# Stage 2: runtime — copy venv + source, run as non-root
# ============================================================

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first (layer-cache friendly)
COPY pyproject.toml README.md LICENSE ./
COPY lexisearch/__init__.py lexisearch/

# Create venv and install the package + API/CLI extras
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip wheel
RUN pip install fastapi uvicorn[standard] click

# Install lexisearch itself (editable for layer caching, then full copy below)
RUN pip install -e ".[all]"

# Copy full source
COPY . .
RUN pip install -e ".[all]"


# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="LexiSearch"
LABEL org.opencontainers.image.description="Production-ready RAG framework API"
LABEL org.opencontainers.image.source="https://github.com/get2salam/lexisearch"
LABEL org.opencontainers.image.licenses="MIT"

# Non-root user for security
RUN groupadd --gid 1001 lexisearch \
    && useradd --uid 1001 --gid 1001 --no-create-home lexisearch

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY --from=builder /build /app

# Ensure non-root user owns the app directory
RUN chown -R lexisearch:lexisearch /app

USER lexisearch

# Default environment
ENV LEXISEARCH_EMBEDDER=mock \
    LEXISEARCH_LLM=mock \
    LEXISEARCH_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start with uvicorn
CMD ["uvicorn", "lexisearch.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
