import os, sys, platform, streamlit as st
st.title("🔧 Diagnóstico")
st.write({
  "python": sys.version,
  "platform": platform.platform(),
  "railway_sha": os.getenv("RAILWAY_GIT_COMMIT_SHA","?"),
})
st.success("Si ves esto, el frontend JS funciona y el WebSocket está ok.")
