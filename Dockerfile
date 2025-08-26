FROM python:3.11-slim

# Paquetes base (compilación y fuentes para matplotlib)
RUN apt-get update && apt-get install -y \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código
COPY . .

# Puerto estándar de Spaces
ENV PORT=7860
EXPOSE 7860

# Lanzar Streamlit
CMD ["streamlit", "run", "app_web2.py", "--server.port=7860", "--server.address=0.0.0.0"]
