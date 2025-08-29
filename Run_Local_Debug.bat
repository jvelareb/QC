@echo off
setlocal EnableDelayedExpansion

set VENV=.venv_debug
set APP=app_web2.py

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python no esta en PATH. Instala Python 3.10+.
  pause
  exit /b 1
)

if not exist "%VENV%\Scripts\python.exe" (
  python -m venv "%VENV%"
)
call "%VENV%\Scripts\activate.bat"

python -m pip install --upgrade pip
python -m pip install streamlit numpy matplotlib plotly qiskit qiskit-aer

echo [INFO] Lanzando en local con logs (streamlit_debug.log)...
python - << "PYCODE"
import os, socket, subprocess, time, sys

def free_port(start=8501, end=8600):
    for p in range(start, end+1):
        try:
            with socket.create_connection(("127.0.0.1", p), timeout=0.2):
                pass
        except OSError:
            return p
    raise SystemExit("No free port found")

port = free_port()
with open("streamlit_debug.log","w",encoding="utf-8") as log:
    cmd = [sys.executable,"-m","streamlit","run","app_web2.py","--server.port",str(port),"--server.headless","true"]
    print("[CMD]", " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=log, stderr=log)
    print(f"[OK] Abre http://127.0.0.1:{port}")
    # Espera 60s para que puedas ver el mensaje
    for _ in range(60):
        time.sleep(1)
PYCODE
