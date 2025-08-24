import streamlit as st

# Debe ser la PRIMERA llamada de Streamlit en toda la app
st.set_page_config(page_title="Quantum Toolkit (Secure)", page_icon="🔒", layout="wide")
st.set_option("client.showErrorDetails", True)

# ── Login ───────────────────────────────────────────────────────────────
from auth import login
role, _ = login()
if role is None:
    st.stop()

# ── Carga de la app real con “airbag” ───────────────────────────────────
try:
    import app_web2  # noqa: F401 (se ejecuta al importar)
except Exception as e:
    st.error("❌ Error al renderizar la app.")
    st.exception(e)
    st.stop()
