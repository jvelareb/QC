import os, hashlib
import streamlit as st

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _check(user: str, pwd: str, user_env: str, pass_env_hash: str) -> bool:
    return (user == os.getenv(user_env, "")) and (_sha(pwd) == os.getenv(pass_env_hash, ""))

def login():
    """
    Muestra un formulario de login con roles:
    - user: entra con usuario/contraseña y usa SERVICE_OPENAI_API_KEY (no se le pide la key).
    - admin: además de usuario/contraseña, debe introducir una API key válida (se comprueba formato y/o contra ADMIN_API_KEY si decides exigir exactitud).
    Devuelve (role, openai_key_para_admin_o_None)
    """
    with st.sidebar:
        st.header("🔒 Acceso")
        role = st.radio("Rol", ["user", "admin"], horizontal=True)

        username = st.text_input("Usuario", value="", autocomplete="username")
        password = st.text_input("Contraseña", value="", type="password")
        api_key = None

        if role == "admin":
            st.caption("El modo admin requiere introducir una API key propia.")
            api_key = st.text_input("OpenAI API key (solo admin)", value="", type="password", help="Se validará el formato 'sk-...' u otros formatos actuales.")

        if st.button("Entrar", type="primary", use_container_width=True):
            if role == "user":
                if _check(username, password, "USER_USERNAME", "USER_PASSWORD_SHA256"):
                    st.session_state["auth_role"] = "user"
                    return "user", None
                st.error("Credenciales de usuario no válidas.")
            else:
                # Admin: validar user/pwd
                if not _check(username, password, "ADMIN_USERNAME", "ADMIN_PASSWORD_SHA256"):
                    st.error("Credenciales admin no válidas.")
                    st.stop()

                # Validar API key (formato básico: empieza por 'sk-' o longitud razonable)
                if not api_key or (not api_key.startswith("sk-") and len(api_key) < 20):
                    st.error("API key con formato no válido.")
                    st.stop()

                # (Opcional) Exigir coincidencia con una clave concreta almacenada:
                # Si defines ADMIN_API_KEY en Railway, descomenta para exigirla exacta.
                expected = os.getenv("ADMIN_API_KEY", "")
                if expected:
                    if api_key != expected:
                        st.error("La API key no coincide con ADMIN_API_KEY.")
                        st.stop()

                st.session_state["auth_role"] = "admin"
                return "admin", api_key

        # Si ya había sesión
        if st.session_state.get("auth_role") in ("user", "admin"):
            role = st.session_state["auth_role"]
            return role, None

    # No autenticado aún
    st.info("Usa el panel lateral para iniciar sesión.")
    return None, None
