# Quantum Toolkit (Streamlit)

Aplicación para visualización de la esfera de Bloch, aplicación de puertas cuánticas y circuitos de Qiskit.

## Despliegue local

1. Instalar dependencias: `pip install -r requirements.txt`
2. Ejecutar: `streamlit run app_web2.py`

Para autenticación, establecer variables de entorno APP_USER y APP_PASS.

## Despliegue en Hugging Face Spaces

Usar Dockerfile y runtime.txt para Python 3.11.

## Notas
- Pestañas: Esfera de Bloch, Puertas 1 qubit, Circuitos predefinidos, Editor de código Qiskit.
- Dependencias opcionales: Qiskit para pestañas 3 y 4.