import os
import sys
import time
import socket
import subprocess

# Fuerza a PyInstaller a incluir estos paquetes en el bundle,
# aunque se carguen luego dinámicamente desde app_web2.py
try:
    import streamlit as _st  # noqa: F401
except Exception:
    pass
for _pkg in ("numpy", "matplotlib", "plotly", "qiskit", "qiskit_aer"):
    try:
        __import__(_pkg)
    except Exception:
        pass

import webview  # pywebview para ventana nativa

APP_FILE = "app_web2.py"
PORT = 8555
HOST = "127.0.0.1"


def _app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _wait_port(host: str, port: int, timeout: float = 45.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def main():
    base = _app_dir()
    app_path = os.path.join(base, APP_FILE)

    if not os.path.exists(app_path):
        raise SystemExit(f"No se encuentra {APP_FILE} en {base}")

    # Variables de entorno útiles para Streamlit en modo "headless"
    env = os.environ.copy()
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env.setdefault("BROWSER", "none")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # Ejecutable de Python (dentro del bundle si está congelado)
    python_exec = sys.executable

    cmd = [
        python_exec,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    # Lanza Streamlit en background
    proc = subprocess.Popen(cmd, cwd=base, env=env)

    # Espera a que el servidor esté disponible
    if not _wait_port(HOST, PORT, timeout=60):
        try:
            proc.terminate()
        except Exception:
            pass
        raise SystemExit("No se pudo iniciar el servidor local de la aplicacion.")

    # Abre ventana nativa que embebe la web local
    url = f"http://{HOST}:{PORT}"
    webview.create_window("Quantum Toolkit", url, width=1200, height=820)
    webview.start()

    # Al cerrar la ventana, apagamos el servidor
    try:
        proc.terminate()
    except Exception:
        pass


if __name__ == "__main__":
    main()
