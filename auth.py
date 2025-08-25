import os, hashlib
import streamlit as st

def _sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def _get_env(name: str) -> str:
    return (os.getenv(name, "") or "").strip()

def _check(user: str, pwd: str, user_env: str, pass_env: str) -> bool:
    expected_user = _get_env(user_env)
    expected_hash = _get_env(pass_env)
    return (user.strip() == expected_user) and (_sha256(pwd).lower() == expected_hash.lower())

def login():
    """
    Barra lateral de acceso:
    - user: usuario/contraseña (sin API key)
    - admin: requiere API key con formato 'sk-...' y opcionalmente coincide con ADMIN_API_KEY
    Devuelve (role, api_key) o (None, None)
    """
    with st.sidebar:
        st.header("🔒 Acceso")
        role = st.radio("Rol", ["user", "admin"], horizontal=True)
        username = st.text_input("Usuario", autocomplete="username")
        password = st.text_input("Contraseña", type="password")

        api_key = None
        if role == "admin":
            st.caption("Admin requiere una OpenAI API key propia.")
            api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-...")

        if st.button("Entrar", type="primary", use_container_width=True):
            if role == "user":
                if _check(username, password, "USER_USERNAME", "USER_PASSWORD_SHA256"):
                    st.session_state["auth_role"] = "user"
                    return "user", None
                st.error("❌ Usuario o contraseña incorrectos.")
            else:
                if not _check(username, password, "ADMIN_USERNAME", "ADMIN_PASSWORD_SHA256"):
                    st.error("❌ Credenciales admin incorrectas.")
                    st.stop()
                if not api_key or (not api_key.startswith("sk-") and len(api_key) < 20):
                    st.error("API key inválida.")
                    st.stop()
                expected_admin_key = _get_env("ADMIN_API_KEY")
                if expected_admin_key and api_key != expected_admin_key:
                    st.error("La API key no coincide con ADMIN_API_KEY.")
                    st.stop()
                st.session_state["auth_role"] = "admin"
                return "admin", api_key

        if st.session_state.get("auth_role") in ("user", "admin"):
            role = st.session_state["auth_role"]
            return role, (api_key if role == "admin" else None)

    st.info("👉 Usa el panel lateral para iniciar sesión.")
    return None, None
