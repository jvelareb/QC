# Computación Cuántica — Repositorio del Libro

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929C4?logo=ibm)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)

Repositorio de código, visualizaciones y simulaciones interactivas que acompaña al libro **"Computación Cuántica: Fundamentos, Algoritmos y Aplicaciones"**.

---

## Estructura del Repositorio

```
quantum-book/
├── README.md
├── requirements.txt
├── environment.yml
├── setup.py
│
├── src/                        # Módulos reutilizables
│   ├── __init__.py
│   ├── quantum_gates.py        # Puertas cuánticas básicas
│   ├── quantum_circuits.py     # Construcción de circuitos
│   ├── quantum_math.py         # Álgebra lineal y formalismo cuántico
│   ├── visualization.py        # Funciones de visualización unificadas
│   └── bloch_simulator.py      # Simulador de esfera de Bloch
│
├── notebooks/
│   ├── ch01_introduccion/
│   │   ├── 01_qubits_y_estados.ipynb
│   │   └── 02_puertas_basicas.ipynb
│   ├── ch02_entrelazamiento/
│   │   ├── 01_bell_states.ipynb
│   │   └── 02_teleportacion.ipynb
│   ├── ch03_algoritmos_clasicos/
│   │   ├── 01_deutsch_jozsa.ipynb
│   │   ├── 02_bernstein_vazirani.ipynb
│   │   └── 03_simon.ipynb
│   ├── ch04_fourier_cuantica/
│   │   └── 01_qft.ipynb
│   ├── ch05_busqueda/
│   │   └── 01_grover.ipynb
│   ├── ch06_factorizacion/
│   │   └── 01_shor.ipynb
│   ├── ch07_variacional/
│   │   ├── 01_vqe.ipynb
│   │   └── 02_qaoa.ipynb
│   └── ch08_machine_learning/
│       ├── 01_qsvm.ipynb
│       └── 02_qnn.ipynb
│
├── simulators/
│   ├── bloch_sphere/
│   │   ├── app.py              # Aplicación Streamlit interactiva
│   │   └── bloch_widget.py     # Widget ipywidgets para notebooks
│   └── circuit_builder/
│       └── app.py              # Constructor visual de circuitos
│
├── assets/
│   ├── figures/                # Figuras exportadas
│   └── styles/
│       └── notebook_style.css  # Estilo CSS uniforme para notebooks
│
├── tests/
│   ├── test_gates.py
│   ├── test_algorithms.py
│   └── test_math.py
│
└── docs/
    ├── arquitectura.md
    ├── guia_instalacion.md
    └── referencias.md
```

---

## Instalación Rápida

### Con Conda (recomendado)

```bash
conda env create -f environment.yml
conda activate quantum-book
jupyter lab
```

### Con pip

```bash
pip install -r requirements.txt
jupyter lab
```

---

## Contenido por Capítulos

| Capítulo | Tema | Algoritmos / Conceptos |
|---|---|---|
| 1 | Qubits y puertas | Superposición, medida, H, X, Y, Z, S, T |
| 2 | Entrelazamiento | Bell states, teleportación cuántica |
| 3 | Algoritmos oráculo | Deutsch-Jozsa, Bernstein-Vazirani, Simon |
| 4 | Fourier cuántica | QFT, estimación de fase |
| 5 | Búsqueda | Algoritmo de Grover |
| 6 | Factorización | Algoritmo de Shor |
| 7 | Variacional | VQE, QAOA |
| 8 | Machine Learning | QSVM, QNN |

---

## Simulador Interactivo de Esfera de Bloch

```bash
cd simulators/bloch_sphere
streamlit run app.py
```

---

## Contribuciones y Extensiones

Consulta [`docs/arquitectura.md`](docs/arquitectura.md) para la guía de extensión del repositorio con nuevos capítulos, algoritmos o simuladores.

---

## Licencia

MIT License — libre para uso docente y académico.

---

## Referencia del Libro

> Velasco, J. (2025). *Computación Cuántica: Fundamentos, Algoritmos y Aplicaciones*. [Editorial].
