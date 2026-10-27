FROM python:3.11-slim

# Metadane i konfiguracja środowiska wykonawczego Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

WORKDIR /app

# Instalacja podstawowych narzędzi systemowych do healthchecku i bibliotek graficznych
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Optymalizacja instalacji PyTorch (wersja CPU znacząco redukuje rozmiar obrazu)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Instalacja pozostałych bibliotek projektu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie kodu aplikacji i modelu
COPY . .

# Konfiguracja użytkownika bez uprawnień roota dla bezpieczeństwa
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
