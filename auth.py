import os
import hashlib
import streamlit as st
def _sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()
def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()

def check_plain(user: str, pwd: str, env_user: str, env_pass: str) -> bool:
    return (user or "").strip() == _env(env_user) and (pwd or "") == _env(env_pass)

def login_box():
    st.sidebar.header("🔒 Acceso")
    role = st.sidebar.radio("Rol", ["user", "admin"], horizontal=True)
    u = st.sidebar.text_input("Usuario", key="lg_user")
    p = st.sidebar.text_input("Contraseña", type="password", key="lg_pass")

    if st.sidebar.button("Entrar", use_container_width=True):
        if role == "user":
            ok = check_plain(u, p, "APP_USER", "APP_PASS")
            if ok:
                st.session_state["auth_role"] = "user"
            else:
                st.sidebar.error("Credenciales de usuario no válidas.")
        else:
            ok = check_plain(u, p, "APP_ADMIN", "APP_ADMIN_PASS")
            if ok:
                st.session_state["auth_role"] = "admin"
            else:
                st.sidebar.error("Credenciales de admin no válidas.")

    return st.session_state.get("auth_role")

