import os, hashlib
import streamlit as st

st.set_page_config(page_title="QC Qiskit Minimal", layout="wide", page_icon="⚛️")

def _sha(s: str) -> str: return hashlib.sha256((s or "").encode("utf-8")).hexdigest()
def _get(name: str) -> str: return (os.getenv(name) or "").strip()

def _check(u_in: str, p_in: str, UENV: str, PENV: str) -> bool:
    return (u_in or "").strip() == _get(UENV) and _sha(p_in).lower() == _get(PENV).lower()

def login_box():
    with st.sidebar:
        st.header("🔒 Acceso")
        role = st.radio("Rol", ["user","admin"], horizontal=True)
        u = st.text_input("Usuario", value="")
        p = st.text_input("Contraseña", type="password", value="")
        api = None
        if role == "admin":
            st.caption("Admin puede exigir API key (opcional).")
            api = st.text_input("ADMIN_API_KEY", type="password", value="")
        go = st.button("Entrar", use_container_width=True, type="primary")

    if go:
        if role == "user":
            if _check(u,p,"USER_USERNAME","USER_PASSWORD_SHA256"):
                st.session_state.role = "user"
            else:
                st.sidebar.error("Credenciales de usuario no válidas.")
        else:
            if not _check(u,p,"ADMIN_USERNAME","ADMIN_PASSWORD_SHA256"):
                st.sidebar.error("Credenciales admin no válidas.")
            else:
                exp = _get("ADMIN_API_KEY")
                if exp and api != exp:
                    st.sidebar.error("La API key no coincide con ADMIN_API_KEY.")
                else:
                    st.session_state.role = "admin"
    return st.session_state.get("role")

def main():
    st.caption("🫀 servidor OK - build activo")
    role = login_box()
    if not role:
        st.info("Inicia sesión en el panel lateral.")
        return
    st.success(f"Sesión iniciada como **{role}**")
    import app_web2  # carga la app real

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("❌ La aplicación ha fallado al cargar.")
        st.exception(e)
