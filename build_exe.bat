@echo off
setlocal EnableDelayedExpansion

REM --- Configuración ---
set VENV=.venv_build
set APP=app_web2.py
set MAIN=desktop.py
set EXE_NAME=QuantumToolkit

REM --- Comprobaciones ---
if not exist "%APP%" (
  echo [ERROR] No se encuentra %APP% en esta carpeta.
  exit /b 1
)
if not exist "%MAIN%" (
  echo [ERROR] No se encuentra %MAIN% en esta carpeta.
  exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python no esta en PATH. Instala Python 3.10+.
  exit /b 1
)

REM --- Crear venv limpio para el build ---
if exist "%VENV%" rmdir /s /q "%VENV%"
python -m venv "%VENV%"
call "%VENV%\Scripts\activate.bat"

REM --- Instalar dependencias para la app y para el empaquetado ---
python -m pip install --upgrade pip
REM Si tienes requirements.txt, descomenta la linea siguiente:
REM python -m pip install -r requirements.txt

python -m pip install streamlit pywebview pyinstaller
python -m pip install numpy matplotlib plotly qiskit qiskit-aer

REM --- Limpiar builds anteriores ---
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist "%EXE_NAME%.spec" del /f /q "%EXE_NAME%.spec"

REM --- Empaquetar en un unico .exe ---
pyinstaller ^
  --noconsole ^
  --onefile ^
  --name "%EXE_NAME%" ^
  --add-data "%APP%;." ^
  --hidden-import streamlit ^
  --hidden-import numpy ^
  --hidden-import matplotlib ^
  --hidden-import plotly ^
  --hidden-import qiskit ^
  --hidden-import qiskit_aer ^
  "%MAIN%"

echo.
echo [OK] Hecho. Ejecutable en: dist\%EXE_NAME%.exe
echo.

REM --- Opcional: mantener venv para rebuilds; comenta para borrar ---
REM deactivate
REM rmdir /s /q "%VENV%"

pause
