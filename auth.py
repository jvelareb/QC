import os, hashlib
import streamlit as st

def _sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def _get_env_clean(name: str) -> str:
    """Lee variable de entorno, quita espacios y minúsculas para hashes."""
    v = os.getenv(name, "") or ""
    return v.strip()

def _get_hash_with_fallback(env_name: str) -> str:
    """Devuelve el hash esperado (limpio). Acepta el fallo de tipeo PASSWOR→PASSWORD."""
    v = _get_env_clean(env_name)
    if not v and "PASSWORD" in env_name:
        v = _get_env_clean(env_name.replace("PASSWORD", "PASSWOR"))  # fallback por deploy viejo
    return v.lower()

def _check(user: str, pwd: str, user_env: str, pass_env_hash: str) -> bool:
    expected_user = _get_env_clean(user_env)
    expected_hash = _get_hash_with_fallback(pass_env_hash)

    given_user = (user or "").strip()
    given_hash = _sha(pwd).lower()

    if os.getenv("DEBUG_AUTH", "") == "1":
        with st.sidebar:
            st.caption("🔎 DEBUG AUTH")
            st.write({
                "user_env": user_env,
                "expected_user": expected_user,
                "pass_env": pass_env_hash,
                "expected_hash_last8": expected_hash[-8:],
                "given_user": given_user,
                "given_hash_last8": given_hash[-8:]
            })

    return (given_user == expected_user) and (given_hash == expected_hash)

def login():
    with st.sidebar:
        st.header("🔒 Acceso")
        role = st.radio("Rol", ["user", "admin"], horizontal=True)

        username = st.text_input("Usuario", value="", autocomplete="username")
        password = st.text_input("Contraseña", value="", type="password")

        api_key = None
        if role == "admin":
            st.caption("El modo admin requiere una OpenAI API key propia.")
            api_key = st.text_input("OpenAI API key (solo admin)", value="", type="password")

        if st.button("Entrar", type="primary", use_container_width=True):
            if role == "user":
                if _check(username, password, "USER_USERNAME", "USER_PASSWORD_SHA256"):
                    st.session_state["auth_role"] = "user"
                    return "user", None
                st.error("Credenciales de usuario no válidas.")
            else:
                # Admin: validar credenciales y API key
                if not _check(username, password, "ADMIN_USERNAME", "ADMIN_PASSWORD_SHA256"):
                    st.error("Credenciales admin no válidas.")
                    st.stop()

                # Validación simple de formato de key
                if not api_key or (not api_key.startswith("sk-") and len(api_key) < 20):
                    st.error("API key con formato no válido.")
                    st.stop()

                expected_admin_key = _get_env_clean("ADMIN_API_KEY")
                if expected_admin_key and api_key != expected_admin_key:
                    st.error("La API key no coincide con ADMIN_API_KEY.")
                    st.stop()

                st.session_state["auth_role"] = "admin"
                return "admin", api_key

        # Sesión ya autenticada
        if st.session_state.get("auth_role") in ("user", "admin"):
            role = st.session_state["auth_role"]
            return role, None

    # No autenticado aún
    st.info("Usa el panel lateral para iniciar sesión.")
    return None, None

