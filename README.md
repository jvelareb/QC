# Quantum Toolkit (Secure on Railway)

App Streamlit con login (user/admin). 
- Rol `user`: entra con usuario/contraseña y usa SERVICE_OPENAI_API_KEY.
- Rol `admin`: además requiere introducir su propia OpenAI API key (o una que coincida con ADMIN_API_KEY si la defines).

## Ejecutar local
python -m venv .venv
. .venv/Scripts/activate   # Windows
pip install -r requirements.txt
set SERVICE_OPENAI_API_KEY=sk-XXXX...    # Windows (o export en Linux/Mac)
set USER_USERNAME=user
set USER_PASSWORD_SHA256=<hash>
streamlit run secure_app.py

## Producción (Railway)
Start command en Procfile. Añade variables de entorno indicadas en la guía del repo.
