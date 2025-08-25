# QC (Streamlit + Qiskit) — Railway (Docker)

## Despliegue
1. Conecta el repo a Railway → detectará el **Dockerfile** automáticamente.
2. Variables de entorno (Service → Variables):
   - `USER_USERNAME`
   - `USER_PASSWORD_SHA256`
   - `ADMIN_USERNAME`
   - `ADMIN_PASSWORD_SHA256`
   - (opcional) `ADMIN_API_KEY`
   - (opcional) `OPENAI_API_KEY`

Generar hash:
```python
import hashlib; print(hashlib.sha256("TU_CONTRASEÑA".encode()).hexdigest())

