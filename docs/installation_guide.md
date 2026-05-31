# Installation Guide

## System requirements

- Python 3.10 or higher
- Conda (recommended) or pip
- 4 GB RAM minimum (8 GB recommended for simulations with more than 12 qubits)

## Option 1: Conda (recommended)

```bash
git clone https://github.com/<usuario>/quantum-book.git
cd quantum-book
conda env create -f environment.yml
conda activate quantum-book
jupyter lab
```

## Option 2: pip with venv

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

## Launch the Bloch simulator

```bash
conda activate quantum-book
cd simulators/bloch_sphere
streamlit run app.py
```

## Run the tests

```bash
pytest tests/ -v
```

## Common troubleshooting

### Error: `qiskit_aer` not found

```bash
pip install qiskit-aer
```

### Error: `ipywidgets` not visible in JupyterLab

```bash
pip install ipywidgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Qiskit deprecation warnings

Qiskit 1.x has introduced API changes. If you see warnings about `QuantumCircuit.bind_parameters`, update to the latest version:

```bash
pip install --upgrade qiskit qiskit-aer qiskit-algorithms
```
