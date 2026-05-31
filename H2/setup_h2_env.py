
import os
import subprocess
import sys
import venv
from pathlib import Path

def create_h2_environment():
    """
    Crea un entorno virtual y carga las dependencias necesarias para el caso H2.
    Diseñado para entornos Linux/WSL.
    """
    env_name = ".venv_h2"
    env_dir = Path.cwd() / env_name
    
    print(f"--- Configurando el entorno para el caso H2 en: {env_dir} ---")
    
    # 1. Crear el entorno virtual
    if not env_dir.exists():
        print(f"[*] Creando entorno virtual: {env_name}...")
        venv.create(env_dir, with_pip=True)
    else:
        print(f"[!] El entorno {env_name} ya existe. Saltando creación.")

    # 2. Determinar la ruta del ejecutable de Python en el entorno
    # En WSL/Linux suele estar en bin/python
    python_bin = env_dir / "bin" / "python"
    if not python_bin.exists():
        # Por si acaso fuera Windows (aunque estamos en WSL)
        python_bin = env_dir / "Scripts" / "python.exe"

    if not python_bin.exists():
        print("Error: No se ha podido localizar el ejecutable de Python en el entorno.")
        return

    # 3. Actualizar pip
    print("[*] Actualizando pip...")
    subprocess.run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"])

    # 4. Instalar paquetes necesarios
    # Se especifican versiones compatibles para evitar conflictos en el cálculo VQE
    dependencies = [
        "qiskit>=1.0.0",
        "qiskit-nature[pyscf]>=0.7.2",
        "qiskit-algorithms",
        "pyscf",
        "matplotlib",
        "pandas",
        "numpy"
    ]
    
    print(f"[*] Instalando dependencias científicas: {', '.join(dependencies)}...")
    subprocess.run([str(python_bin), "-m", "pip", "install"] + dependencies)

    print("\n" + "="*60)
    print("¡ENTORNO CONFIGURADO CON ÉXITO!")
    print("="*60)
    print(f"\nPara activar el entorno, ejecute:")
    print(f"source {env_name}/bin/activate")
    print(f"\nDespués podrá ejecutar sus simulaciones de la molécula H2.")
    print("="*60)

if __name__ == "__main__":
    create_h2_environment()
