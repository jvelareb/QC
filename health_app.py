import streamlit as st, sys, platform, os
st.set_page_config(page_title="Health", page_icon="✅")
st.title("✅ Streamlit OK en Docker")
st.write({"python": sys.version, "platform": platform.platform(), "cwd": os.getcwd()})
st.success("Si ves esto, la plataforma/render funciona.")
