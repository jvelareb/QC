# ====== Base ======
FROM python:3.12-slim

# Para que matplotlib no intente usar backends de GUI
ENV MPLBACKEND=Agg
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Streamlit headless
ENV STREAMLIT_SERVER_HEADLESS=true

# ====== Dependencias del sistema que suelen faltar ======
# - libglib2.0/mesa/libgl1: evitan errores de render de mpl/plotly en contenedores
# - fonts-dejavu: para que el texto no salga “vacío” en figuras
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libglib2.0-0 \
    libgl1 \
    fonts-dejavu-core \
  && rm -rf /var/lib/apt/lists/*

# ====== Workdir ======
WORKDIR /app

# ====== Python deps ======
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# ====== Copia del proyecto ======
COPY . /app

# ====== Exponer y lanzar ======
# Railway inyecta $PORT. No uses valor fijo, pero exponer 8501 ayuda localmente.
EXPOSE 8501

# HEALTHCHECK para que Railway sepa si está vivo.
# Streamlit responde 200 en la raíz cuando la app arranca.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/ || exit 1

# IMPORTANTE: usa ${PORT} (no "$PORT") para que se expanda antes de ejecutar
CMD streamlit run app_web2.py --server.address=0.0.0.0 --server.port=${PORT}


