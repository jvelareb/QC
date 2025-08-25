# Quantum Toolkit (Streamlit + Qiskit, login)

App de 4 pestañas:
1) Esfera de Bloch (estado 1-qubit)
2) Puertas 1-qubit (in/out)
3) Circuitos Qiskit predeterminados (carga lazy)
4) Editor Qiskit (carga lazy)

## Despliegue en Hugging Face Spaces

1. Crea un Space con:
   - SDK: **Streamlit**
   - Python: **3.11**
   - **Main file**: `app_web2.py`
2. Sube `app_web2.py`, `auth.py`, `requirements.txt`.
3. En **Settings → Variables**, define:
   - `APP_USER` = `usuario`
   - `APP_PASS` = `tu_clave`
   - `APP_ADMIN` = `admin`
   - `APP_ADMIN_PASS` = `otra_clave`
4. Pulsa **Deploy**.

## Notas técnicas
- Qiskit se importa bajo demanda (botón) para evitar timeouts al iniciar.
- Export de figuras cacheado vía `@st.cache_data` con parámetro `_fig` (evita UnhashableParamError).
- Para diagramas con `circuit_drawer(output="mpl")` se requiere `pylatexenc`.
