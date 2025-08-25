# secure_app.py
import os, hashlib
import streamlit as st

st.set_page_config(page_title="QC Toolkit", page_icon="⚛️", layout="wide")

# ---------- utilidades auth ----------
def _sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def _get_env(name: str) -> str:
    return (os.getenv(name, "") or "").strip()

def _get_expected_hash(env_name: str) -> str:
    v = _get_env(env_name)
    # fallback por posible typo PASSWOR vs PASSWORD en variables antiguas
    if not v and env_name.endswith("PASSWORD_SHA256"):
        v = _get_env(env_name.replace("PASSWORD", "PASSWOR"))
    return v.lower()

def _check(user: str, pwd: str, user_env: str, pass_env_hash: str) -> bool:
    expected_user = _get_env(user_env)
    expected_hash = _get_expected_hash(pass_env_hash)
    given_user = (user or "").strip()
    given_hash = _sha(pwd).lower()
    return (given_user == expected_user) and (given_hash == expected_hash)

# ---------- UI login ----------
with st.sidebar:
    st.header("🔒 Acceso")
    role = st.radio("Rol", ["user", "admin"], horizontal=True)

    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")

    api_key = None
    if role == "admin":
        st.caption("Admin requiere su propia OpenAI API key (opcional validar).")
        api_key = st.text_input("OpenAI API key (solo admin)", type="password", placeholder="sk-...")

    login_click = st.button("Entrar", type="primary", use_container_width=True)

if login_click:
    if role == "user":
        if _check(username, password, "USER_USERNAME", "USER_PASSWORD_SHA256"):
            st.session_state["auth_role"] = "user"
        else:
            st.error("Credenciales de usuario no válidas.")
    else:
        if not _check(username, password, "ADMIN_USERNAME", "ADMIN_PASSWORD_SHA256"):
            st.error("Credenciales admin no válidas.")
        else:
            # Validación simple de formato
            if api_key and not api_key.startswith("sk-"):
                st.error("Formato de API key no válido (debería empezar por 'sk-').")
            expected = _get_env("ADMIN_API_KEY")
            if expected and api_key and api_key != expected:
                st.error("La API key no coincide con ADMIN_API_KEY.")
            else:
                st.session_state["auth_role"] = "admin"
                if api_key:
                    st.session_state["OPENAI_API_KEY"] = api_key  # por si tu app la usa

# Estado actual
role_ss = st.session_state.get("auth_role")

if role_ss in ("user", "admin"):
    st.sidebar.success(f"Sesión iniciada como **{role_ss}**")
    # Importa y ejecuta tu app (se evalúa al importar)
    import app_web2  # noqa: F401
else:
    st.title("⚛️ QC Toolkit")
    st.info("Inicia sesión en el panel lateral para continuar.")
    st.caption("Roles: **user** (sin API key) | **admin** (pide API key).")
