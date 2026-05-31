# Guía de Instalación

## Requisitos del sistema

- Python 3.10 o superior
- Conda (recomendado) o pip
- 4 GB de RAM mínimo (8 GB recomendado para simulaciones de más de 12 qubits)

## Opción 1: Conda (recomendado)

```bash
git clone https://github.com/<usuario>/quantum-book.git
cd quantum-book
conda env create -f environment.yml
conda activate quantum-book
jupyter lab
```

## Opción 2: pip con venv

```bash
git clone https://github.com/<usuario>/quantum-book.git
cd quantum-book
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Lanzar el simulador de Bloch

```bash
conda activate quantum-book
cd simulators/bloch_sphere
streamlit run app.py
```

## Ejecutar los tests

```bash
pytest tests/ -v
```

## Solución de problemas comunes

### Error: `qiskit_aer` no encontrado

```bash
pip install qiskit-aer
```

### Error: `ipywidgets` no visible en JupyterLab

```bash
pip install ipywidgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Warnings de deprecación de Qiskit

Qiskit 1.x ha introducido cambios en la API. Si ves warnings sobre `QuantumCircuit.bind_parameters`, actualiza a la última versión:

```bash
pip install --upgrade qiskit qiskit-aer qiskit-algorithms
```
