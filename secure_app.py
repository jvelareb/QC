import os
import streamlit as st
from auth import login

# ✅ Debe ser la PRIMERA llamada de Streamlit en toda la app
st.set_page_config(page_title="Quantum Toolkit (Secure)", page_icon="🔒", layout="wide")
st.set_option("client.showErrorDetails", True)  # ver trazas en UI si algo peta

# ─────────────────────────────
# 1) LOGIN
# ─────────────────────────────
role, provided_key = login()
if role is None:
    st.stop()

# Selección de API key según rol
if role == "user":
    service_key = os.getenv("SERVICE_OPENAI_API_KEY", "").strip()
    if not service_key:
        st.error("Falta SERVICE_OPENAI_API_KEY en variables de entorno.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = service_key
else:  # admin
    if not provided_key:
        st.error("Admin sin API key válida.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = provided_key.strip()

# (Opcional) debug de auth si pones DEBUG_AUTH=1 en Railway
if os.getenv("DEBUG_AUTH", "") == "1":
    with st.sidebar:
        st.caption("🔎 DEBUG ENV")
        st.write({
            "ROLE": role,
            "HAS_OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        })

# ─────────────────────────────
# 2) Cargar tu app real con protección de errores
# ─────────────────────────────
try:
    import app_web2  # noqa: F401  (tu app se ejecuta al importar)
except Exception as e:
    st.error("❌ Error al renderizar la app (ver traza):")
    st.exception(e)
    st.stop()

