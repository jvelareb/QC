# Quantum Computing — Book Repository

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929C4?logo=ibm)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)

Code repository, visualizations, and interactive simulations accompanying the book **"Quantum Computing: Foundations, Algorithms, and Applications"**.

---

## Repository Structure

```
quantum-book/
├── README.md
├── requirements.txt
├── environment.yml
├── setup.py
│
├── src/                        # Reusable modules
│   ├── __init__.py
│   ├── quantum_gates.py        # Basic quantum gates
│   ├── quantum_circuits.py     # Circuit construction
│   ├── quantum_math.py         # Linear algebra and quantum formalism
│   ├── visualization.py        # Unified visualization functions
│   └── bloch_simulator.py      # Bloch sphere simulator
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
│   │   ├── app.py              # Interactive Streamlit application
│   │   └── bloch_widget.py     # ipywidgets widget for notebooks
│   └── circuit_builder/
│       └── app.py              # Visual circuit builder
│
├── assets/
│   ├── figures/                # Exported figures
│   └── styles/
│       └── notebook_style.css  # Uniform CSS style for notebooks
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

## Quick Installation

### With Conda (recommended)

```bash
conda env create -f environment.yml
conda activate quantum-book
jupyter lab
```

### With pip

```bash
pip install -r requirements.txt
jupyter lab
```

---

## Content by Chapter

| Chapter | Topic | Algorithms / Concepts |
|---|---|---|
| 1 | Qubits and gates | Superposition, measurement, H, X, Y, Z, S, T |
| 2 | Entanglement | Bell states, quantum teleportation |
| 3 | Oracle algorithms | Deutsch-Jozsa, Bernstein-Vazirani, Simon |
| 4 | Quantum Fourier | QFT, phase estimation |
| 5 | Search | Grover's algorithm |
| 6 | Factorization | Shor's algorithm |
| 7 | Variational | VQE, QAOA |
| 8 | Machine Learning | QSVM, QNN |

---

## Interactive Bloch Sphere Simulator

```bash
cd simulators/bloch_sphere
streamlit run app.py
```

---

## Contributions and Extensions

See [`docs/arquitectura.md`](docs/arquitectura.md) for the repository extension guide covering new chapters, algorithms, or simulators.

---

## License

MIT License — free for educational and academic use.

---

## Book Reference

> Velasco, J. (2025). *Quantum Computing: Foundations, Algorithms, and Applications*. [Publisher].
