# Imagen ligera y estable para Qiskit + Streamlit
FROM python:3.10-slim

# Evitar bytecode y forzar logs en stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# Dependencias del sistema (necesarias para qiskit-aer y compiladores básicos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala dependencias Python primero (aprovecha cache de capas Docker)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

# Copia el resto del proyecto
COPY . .

# Railway inyecta $PORT. Por si acaso, define un default.
ENV PORT=8080

# Lanza Streamlit escuchando en 0.0.0.0:$PORT y sin CORS/XSRF para evitar la pantalla en blanco
CMD streamlit run app_web2.py \
    --server.headless=true \
    --server.address=0.0.0.0 \
    --server.port=${PORT} \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
