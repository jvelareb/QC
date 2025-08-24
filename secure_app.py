import os
import streamlit as st
from auth import login

st.set_page_config(page_title="Quantum Toolkit (Secure)", page_icon="🔒", layout="wide")

role, provided_key = login()

# Si no ha pasado login, cortar la ejecución.
if role is None:
    st.stop()

# Selección de API key según el rol
if role == "user":
    service_key = os.getenv("SERVICE_OPENAI_API_KEY", "")
    if not service_key:
        st.error("Falta SERVICE_OPENAI_API_KEY en variables de entorno.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = service_key
elif role == "admin":
    os.environ["OPENAI_API_KEY"] = provided_key

# Importar y renderizar tu app (se ejecuta al importar)
import app_web2  # noqa: F401
