# Repository Usage Guide

Welcome to the quantum computing repository that accompanies the book.
This guide explains what each file is for and how to work with it.

---

## General structure

```
./
├── GUIA.md                    ← This file
├── README.md                  ← Repository front page (installation, requirements)
├── requirements.txt           ← pip dependencies
├── environment.yml            ← Full Conda environment
├── setup.py                   ← Installation of the src/ package as a module
│
├── src/                       ← Reusable modules (imported from the notebooks)
├── notebooks/                 ← Pedagogical notebooks organized by chapter
├── docs/                      ← Technical documentation
└── tests/                     ← Automated tests for the src/ modules
```

---

## 1. Getting started

### Environment setup (once only)

**Option A — Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate quantum-book
```

**Option B — pip:**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

### Launch JupyterLab

```bash
jupyter lab
```

---

## 2. Notebooks — chapter-by-chapter guide

All notebooks are in `notebooks/` organized by chapter.
Open them from JupyterLab and run the cells in order.

---

### Chapter 1 · Introduction to Qubits (`ch01_introduccion/`)

#### `01_qubits_y_estados.ipynb`

**What it does?**
Introduces the mathematical representation of a qubit, single-qubit gates
and visualization on the Bloch sphere. This is the recommended entry point
for any reader starting the book.

**Content:**
- Qubit state: amplitudes α and β, probabilities.
- Gates H, X, Y, Z, S, T as unitary matrices.
- Calculation of the Bloch vector.
- Static visualization with matplotlib.

**Modules used:**
`src/quantum_gates.py`, `src/quantum_math.py`, `src/visualization.py`

---

#### `02_bloch_interactivo.ipynb`

**What it does?**
Interactive Bloch sphere simulator with `ipywidgets` controls
directly inside JupyterLab. No external application needed.

**How to use it:**
1. Run `Kernel → Restart & Run All`.
2. A panel appears with controls on the left and the 3D sphere on the right.
3. Choose the initial state, apply fixed or parametric gates with sliders,
   or run predefined sequences.
4. The final cell allows you to write your own gate sequence.

**Modules used:**
`src/bloch_simulator.py`, `src/quantum_math.py`

> **Additional requirement:** `ipywidgets`
> ```bash
> pip install ipywidgets
> ```

---

### Chapter 2 · Entanglement (`ch02_entrelazamiento/`)

#### `01_bell_states.ipynb`

**What it does?**
Builds and analyzes the four Bell states. Introduces quantum entanglement,
Von Neumann entropy and quantum teleportation.

**Content:**
- Creation of Bell states with H + CNOT.
- Measurement of entanglement entropy.
- Quantum teleportation protocol (noiseless simulation).

---

### Chapter 3 · Classical Algorithms (`ch03_algoritmos_clasicos/`)

#### `01_deutsch_jozsa.ipynb`

**What it does?**
Implements the Deutsch-Jozsa algorithm: the first example of exponential
quantum advantage over a deterministic classical algorithm.

**Content:**
- Constant and balanced oracles for n bits.
- Demonstration that 1 quantum query suffices versus 2^(n-1)+1 classical ones.
- Visualization of the circuit and measurements.

---

#### `02_bernstein_vazirani.ipynb`

**What it does?**
Solves the Bernstein-Vazirani problem: finding a hidden string of $n$ bits
with a single oracle query.

**Content:**
- Oracle construction for an arbitrary secret string.
- Verification that the circuit recovers the exact string in a single measurement.

---

#### `03_simon.ipynb`

**What it does?**
Implements Simon's algorithm to find the hidden period of a function.
It is the conceptual foundation of Shor's algorithm.

**Content:**
- Simon's oracle, quantum circuit (double Hadamard application).
- Classical post-processing: linear algebra mod 2.
- Verification that `y · s = 0 (mod 2)` for each measurement.

---

### Chapter 4 · Quantum Fourier Transform (`ch04_fourier_cuantica/`)

#### `01_qft.ipynb`

**What it does?**
Implements the Quantum Fourier Transform (QFT) and verifies its correctness
by comparing it with the analytically derived exact matrix.

**Content:**
- Definition of the QFT and its circuit (H gates and controlled phase CR_k gates).
- Numerical verification: `QFT_Qiskit ≈ QFT_analytic`.
- Complexity analysis: O(n²) gates vs O(n·2ⁿ) of the classical FFT.

---

### Chapter 5 · Quantum Search (`ch05_busqueda/`)

#### `01_grover.ipynb`

**What it does?**
Implements Grover's algorithm to search for an element in a database.
Quadratic speedup over classical search: O(√N) vs O(N).

**Content:**
- Grover Multi-Controlled-Z (MCZ) oracle.
- Grover diffusion operator (inversion about the mean).
- Optimal number of iterations: π√N/4.
- Probability plot of the marked item as a function of iterations.

---

### Chapter 6 · Factorization (`ch06_factorizacion/`)

#### `01_shor.ipynb`

**What it does?**
Implements Shor's algorithm for N = 15 (pedagogical case).
Combines quantum phase estimation (QPE) with the classical continued fractions algorithm.

**Content:**
- Algorithm structure: QPE + classical reduction to GCD.
- Full circuit for N = 15, a = 2 (period r = 4).
- Factor verification: 15 = 3 × 5.

---

### Chapter 7 · Variational Algorithms (`ch07_variacional/`)

#### `01_vqe.ipynb`

**What it does?**
Implements VQE (Variational Quantum Eigensolver) to find the ground state
energy of the 2-qubit Heisenberg Hamiltonian.

**Content:**
- Definition of the Heisenberg Hamiltonian as SparsePauliOp.
- RealAmplitudes ansatz, COBYLA optimizer.
- Energy convergence curve.

---

#### `02_qaoa.ipynb`

**What it does?**
Implements QAOA (Quantum Approximate Optimization Algorithm) for the
MaxCUT problem on a 4-node graph.

**Content:**
- Cost Hamiltonian for MaxCUT: H_C = Σ w_ij (I − Z_i Z_j) / 2.
- QAOA ansatz of depth p with cost and mixing layers.
- Optimization with COBYLA and approximation ratio analysis.

---

#### `03_vqe_h2.ipynb`

**What it does?**
Complete VQE demonstration for the hydrogen molecule H₂, based on
real calculations with PySCF and Qiskit Nature.

**Content:**
1. PySCF driver → fermionic Hamiltonian in second quantization.
2. Jordan-Wigner mapping → 4-qubit Hamiltonian (15 Pauli terms).
3. UCCSD ansatz with Hartree-Fock initial state (3 parameters).
4. VQE with SLSQP: convergence to FCI with error < 1 mHa.
5. H₂ dissociation curve: HF, FCI and VQE energies for R = 0.35–2.50 Å.

> **Requires:** `qiskit-nature`, `pyscf`
> ```bash
> pip install qiskit-nature pyscf
> ```

---

### Chapter 8 · Quantum Machine Learning (`ch08_machine_learning/`)

#### `01_qsvm.ipynb`

**What it does?**
Implements a Support Vector Machine with a quantum kernel (QSVM)
and compares it with a classical RBF kernel SVM.

**Content:**
- ZZFeatureMap feature map (2 qubits, reps=2).
- Quantum kernel: K(x, x') = |⟨φ(x)|φ(x')⟩|².
- Classification of the "two moons" dataset with a precomputed SVM.
- Accuracy comparison: quantum vs classical.

> **Requires:** `qiskit-machine-learning`
> ```bash
> pip install qiskit-machine-learning
> ```

---

#### `02_qnn.ipynb`

**What it does?**
Implements a Quantum Neural Network (QNN) with Qiskit's `EstimatorQNN`
for binary classification of concentric circles.

**Content:**
- Architecture: ZZFeatureMap (embedding) + RealAmplitudes (variational).
- Training with COBYLA and `NeuralNetworkClassifier`.
- Learning curve and final accuracy.
- Discussion of the barren plateau problem.

---

## 3. `src/` modules — technical reference

The `src/` modules are the repository's shared library.
**They are not executable scripts**; they are imported from the notebooks with:

```python
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '..', '..'))
from src.quantum_gates import Gates
```

### `src/quantum_gates.py`

`Gates` class with all quantum gates as NumPy `complex128` matrices.

| Method / Attribute | Description |
|---|---|
| `Gates.I`, `Gates.X`, `Gates.Y`, `Gates.Z` | Pauli gates |
| `Gates.H` | Hadamard |
| `Gates.S`, `Gates.T`, `Gates.Sdg`, `Gates.Tdg` | Phase gates |
| `Gates.CNOT`, `Gates.CZ`, `Gates.SWAP` | 2-qubit gates |
| `Gates.Rx(θ)`, `Gates.Ry(θ)`, `Gates.Rz(θ)` | Parametric rotations |
| `Gates.P(φ)` | Phase gate |
| `Gates.CR_k(k)` | Controlled phase rotation (for QFT) |
| `Gates.tensor(*gates)` | Tensor product of multiple gates |
| `Gates.is_unitary(U)` | Verifies U†U = I |
| `Gates.apply(U, state)` | Applies U to the state |

---

### `src/quantum_math.py`

`QuantumMath` class with quantum algebra tools.

| Method | Description |
|---|---|
| `QuantumMath.ket0()`, `QuantumMath.ket1()` | Basis states |
| `QuantumMath.ket_plus()`, `QuantumMath.ket_minus()` | Hadamard states |
| `QuantumMath.tensor_product(s1, s2)` | Tensor product of states |
| `QuantumMath.probabilities(state)` | Measurement probability vector |
| `QuantumMath.inner_product(s1, s2)` | Inner product ⟨s1|s2⟩ |
| `QuantumMath.bloch_vector(state)` | Bloch vector (x, y, z) |
| `QuantumMath.bloch_angles(state)` | Polar angles (θ, φ) |
| `QuantumMath.density_matrix(state)` | Density matrix ρ = |ψ⟩⟨ψ| |
| `QuantumMath.fidelity(s1, s2)` | Fidelity F = \|⟨s1\|s2⟩\|² |
| `QuantumMath.von_neumann_entropy(rho)` | Entropy S = -Tr(ρ log ρ) |
| `QuantumMath.qft_matrix(n)` | Analytic QFT matrix for n qubits |
| `QuantumMath.normalize(state)` | Normalizes a state vector |

---

### `src/visualization.py`

`QuantumVisualization` class with plots in academic dark style.

| Method | Description |
|---|---|
| `plot_histogram(counts)` | Measurement histogram (shots results) |
| `plot_bloch_vector(state)` | Static Bloch sphere (matplotlib) |
| `plot_unitary(U, title)` | Heat map of a unitary matrix |
| `plot_statevector(state)` | Amplitudes and probabilities of the statevector |

---

### `src/bloch_simulator.py`

`BlochSimulator` class for interactive single-qubit simulation.
It is the engine behind the `02_bloch_interactivo.ipynb` notebook.

| Method | Description |
|---|---|
| `BlochSimulator(initial_state)` | Creates the simulator with an initial state |
| `apply_gate(gate, **kwargs)` | Applies a gate and records the position |
| `apply_sequence([{gate,...}])` | Applies a list of gates |
| `plot_trajectory(title, dark_mode)` | 3D Plotly figure with the trajectory |
| `state_info()` | Returns dict with α, β, probabilities, Bloch |
| `reset(initial_state)` | Resets the simulator |

---

## 4. Tests

The tests mathematically verify the `src/` modules.

```bash
# Run from the repository root
pytest tests/ -v
```

| File | What it verifies |
|---|---|
| `tests/test_gates.py` | Gate unitarity, algebraic identities (HXH=Z, T⁸=I, ...) |
| `tests/test_math.py` | Normalization, Bloch vector, fidelity, entropy, unitary QFT |

---

## 5. Code conventions

- **Python**: strict PEP-8. Functions with NumPy-style docstrings.
- **Notebooks**: one cell = one idea. No cells longer than 40 lines.
- **LaTeX**: always in `$$...$$` (display) or `$...$` (inline).
- **Plots**: use `QuantumVisualization` from `src/` to maintain a unified style.
- **Seeds**: always `np.random.seed(42)` at the start of notebooks with randomness.
- **Paths**: always relative to the file's folder (never absolute disk paths).
