import streamlit as st
import hashlib
import os

def _sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def _env(name: str, default: str) -> str:
    return (os.getenv(name, default) or "").strip()

# Usuarios desde Variables (Settings → Variables / Secrets)
USERS = {
    _env("APP_USER", "user"): _sha(_env("APP_PASS", "1234")),
    _env("APP_ADMIN", "admin"): _sha(_env("APP_ADMIN_PASS", "admin123")),
}

def login() -> bool:
    if st.session_state.get("auth_ok"):
        return True

    with st.sidebar:
        st.header("🔒 Login")
        u = st.text_input("Usuario", key="login_user")
        p = st.text_input("Contraseña", type="password", key="login_pass")
        if st.button("Entrar", type="primary", use_container_width=True):
            if u in USERS and USERS[u] == _sha(p):
                st.session_state["auth_ok"] = True
                st.success("✅ Login correcto")
                return True
            st.error("❌ Usuario o contraseña incorrectos")
    return False
