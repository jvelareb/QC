# QC Qiskit Minimal (Streamlit + Railway)

## Variables de entorno (Railway → Variables)
- USER_USERNAME
- USER_PASSWORD_SHA256
- ADMIN_USERNAME
- ADMIN_PASSWORD_SHA256
- ADMIN_API_KEY  (opcional)

### SHA-256 (PowerShell)
$pwd = Read-Host "Contraseña"
$bytes = [Text.Encoding]::UTF8.GetBytes($pwd)
$hash  = [Security.Cryptography.SHA256]::Create().ComputeHash($bytes)
-join ($hash | ForEach-Object { $_.ToString("x2") })

### Despliegue
1) git init && git add . && git commit -m "init qiskit"
2) Sube a GitHub
3) Railway → New Project → Deploy from GitHub
4) Añade variables y Deploy
