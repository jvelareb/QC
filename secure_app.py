import os, hashlib, traceback, importlib
import streamlit as st

st.set_page_config(page_title="QC Toolkit", page_icon="⚛️", layout="wide")

def _sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def _get_env(name: str) -> str:
    return (os.getenv(name, "") or "").strip()

def _get_expected_hash(env_name: str) -> str:
    v = _get_env(env_name)
    if not v and env_name.endswith("PASSWORD_SHA256"):
        v = _get_env(env_name.replace("PASSWORD", "PASSWOR"))  # fallback por typos antiguos
    return v.lower()

def _check(user: str, pwd: str, user_env: str, pass_env_hash: str) -> bool:
    return (user or "").strip() == _get_env(user_env) and _sha(pwd).lower() == _get_expected_hash(pass_env_hash)

with st.sidebar:
    st.header("🔒 Acceso")
    role = st.radio("Rol", ["user", "admin"], horizontal=True)
    u = st.text_input("Usuario")
    p = st.text_input("Contraseña", type="password")
    api_key = None
    if role == "admin":
        st.caption("Admin puede requerir OpenAI API key.")
        api_key = st.text_input("OpenAI API key (opcional)", type="password", placeholder="sk-...")
    go = st.button("Entrar", type="primary", use_container_width=True)

if go:
    if role == "user":
        st.session_state.pop("auth_role", None)
        if _check(u, p, "USER_USERNAME", "USER_PASSWORD_SHA256"):
            st.session_state["auth_role"] = "user"
        else:
            st.error("Credenciales de usuario no válidas.")
    else:
        st.session_state.pop("auth_role", None)
        if not _check(u, p, "ADMIN_USERNAME", "ADMIN_PASSWORD_SHA256"):
            st.error("Credenciales admin no válidas.")
        else:
            exp = _get_env("ADMIN_API_KEY")
            if api_key and not api_key.startswith("sk-"):
                st.error("Formato de API key no válido (debería empezar por 'sk-').")
            elif exp and api_key and api_key != exp:
                st.error("La API key no coincide con ADMIN_API_KEY.")
            else:
                st.session_state["auth_role"] = "admin"
                if api_key:
                    st.session_state["OPENAI_API_KEY"] = api_key

role_ss = st.session_state.get("auth_role")

if role_ss in ("user", "admin"):
    st.sidebar.success(f"Sesión: **{role_ss}**")
    try:
        import app_web2  # si tu app ejecuta al importar
        # Si expusieras una función explícita:
        # mod = importlib.import_module("app_web2"); mod.run_app()
    except Exception as e:
        st.error("❌ La app falló al iniciar. Debajo tienes el traceback:")
        st.exception(e)
        with st.expander("Ver traceback completo"):
            st.code("".join(traceback.format_exc()))
else:
    st.title("⚛️ QC Toolkit")
    st.info("Inicia sesión en el panel lateral para continuar.")
