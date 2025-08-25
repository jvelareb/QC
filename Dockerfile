# Imagen base oficial de Python
FROM python:3.12-slim

# Configuración de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crea y usa un directorio de trabajo
WORKDIR /app

# Copia requirements y los instala
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copia toda tu app
COPY . /app

# Railway asigna automáticamente el puerto en la variable $PORT
EXPOSE 8501

# Ejecuta Streamlit con el puerto correcto
CMD streamlit run app_web2.py --server.port=${PORT} --server.address=0.0.0.0

