# Repository Architecture

This document describes the complete structure of the repository, the role of each
folder, and the detailed content of each Jupyter notebook.

---

## Folder structure

```
./
├── GUIA.md                  ← Quick-start guide (how to install, how to run)
├── README.md                ← Repository landing page
├── requirements.txt         ← pip dependencies
├── environment.yml          ← Reproducible Conda environment
├── setup.py                 ← Installs the src/ package as a Python module
│
├── src/                     ← Reusable code library
│   ├── quantum_gates.py     ← All quantum gates as NumPy matrices
│   ├── quantum_math.py      ← Quantum algebra (Bloch, fidelity, QFT, entropy)
│   ├── bloch_simulator.py   ← Interactive single-qubit simulator (Plotly 3D)
│   └── visualization.py     ← Unified plots with dark academic style
│
├── notebooks/               ← Pedagogical notebooks by chapter
│   ├── ch01_introduccion/
│   ├── ch02_entrelazamiento/
│   ├── ch03_algoritmos_clasicos/
│   ├── ch04_fourier_cuantica/
│   ├── ch05_busqueda/
│   ├── ch06_factorizacion/
│   ├── ch07_variacional/
│   └── ch08_machine_learning/
│
├── docs/                    ← Technical documentation
│   ├── arquitectura.md      ← This file
│   └── guia_instalacion.md  ← Detailed installation instructions
│
└── tests/                   ← Automated tests for the src/ modules
    ├── test_gates.py
    └── test_math.py
```

---

## Design principle

**Separation of concerns.** Mathematical code lives in `src/` and notebooks
only invoke it. If a function in `quantum_math.py` is improved, the change
propagates automatically to all notebooks without touching any of them.

**Pedagogical progression.** Chapters go from the simplest (qubit, single gate)
to the most complex (VQE for H₂, quantum neural networks), following the order
of the book.

**Always relative paths.** No file contains absolute disk paths.
All imports use `os.path.join(os.getcwd(), '..', '..')` to locate `src/`.

---

## `src/` Modules — summary

| Module | Main class | Purpose |
|---|---|---|
| `quantum_gates.py` | `Gates` | Quantum gates as NumPy matrices: H, X, Y, Z, S, T, CNOT, CZ, Rx, Ry, Rz, P, CR_k... |
| `quantum_math.py` | `QuantumMath` | Basic states, probabilities, Bloch vector, fidelity, Von Neumann entropy, analytical QFT |
| `bloch_simulator.py` | `BlochSimulator` | Applies gate sequences to a qubit, records the trajectory, and generates the Plotly 3D figure |
| `visualization.py` | `QuantumVisualization` | Measurement histograms, static Bloch sphere, unitary heat maps, statevectors |

---

## Notebooks — detailed description

---

### Chapter 1 · Introduction to Qubits

#### `ch01_introduccion/01_qubits_y_estados.ipynb`

**Chapter objective:** Present the fundamental ideas of quantum computing:
the qubit as a vector in ℂ², quantum gates as unitary matrices,
and the Bloch sphere as a visualization tool.

**Notebook structure:**

1. **Mathematical representation of the qubit.** A qubit is a vector
   `|ψ⟩ = α|0⟩ + β|1⟩` with `|α|² + |β|² = 1`. The Bloch vector
   (x, y, z) is computed from the amplitudes and visualized on the sphere.

2. **Single-qubit gates.** The gates H, X, Y, Z, S, T are constructed
   as 2×2 matrices and their unitarity is verified with `Gates.is_unitary()`.

3. **Applying gates.** Gate sequences are applied to the state |0⟩
   and the intermediate states are plotted on the Bloch sphere.

4. **Measurement probabilities.** `P(|0⟩) = |α|²` and
   `P(|1⟩) = |β|²` are computed, and a measurement is simulated with `np.random.choice`.

5. **Proposed exercises.** Verify HXH = Z, compute the action of T⁸,
   find the state that maximizes P(|1⟩) after applying Rx(θ).

**Modules used:** `src/quantum_gates.py`, `src/quantum_math.py`, `src/visualization.py`

---

#### `ch01_introduccion/02_bloch_interactivo.ipynb`

**Objective:** Explore the Bloch sphere in a fully interactive way
within JupyterLab, without any external application.

**Notebook structure:**

1. **Control panel (ipywidgets).** An interface is built with:
   - Initial state dropdown: |0⟩, |1⟩, |+⟩, |-⟩, |i⟩, |-i⟩.
   - "Reset" button that resets the simulator to the selected state.
   - Fixed gate dropdown (H, X, Y, Z, S, T, Sdg, Tdg) + "Apply" button.
   - Parametric gate dropdown (Rx, Ry, Rz, P) + angle slider in degrees.
   - Predefined sequence dropdown (Hadamard, teleportation, T⁸ = I...).
   - "Clear trajectory" button that erases the trace without changing the state.

2. **3D Bloch sphere (Plotly).** Updates in real time after each action.
   The trajectory is shown as a red line with markers.
   Hovering over a point displays the gate name and coordinates.

3. **Information panel.** Displays at all times:
   amplitudes α and β, probabilities P(|0⟩) and P(|1⟩),
   Bloch vector (x, y, z) and history of applied gates.

4. **Educational section.** Table of basic states, table of gates with their effect
   on the Bloch sphere, and the mathematical formulas for the Bloch vector.

5. **Free experiment.** Code cell where the student can write
   their own sequence with `apply_sequence([...])` and see the result.

**Modules used:** `src/bloch_simulator.py`, `src/quantum_math.py`

**Requires:** `ipywidgets` (`pip install ipywidgets`)

---

### Chapter 2 · Entanglement

#### `ch02_entrelazamiento/01_bell_states.ipynb`

**Chapter objective:** Introduce quantum entanglement as the most characteristic
property of quantum computing, with no classical analogue.

**Notebook structure:**

1. **The four Bell states.** The states
   |Φ⁺⟩, |Φ⁻⟩, |Ψ⁺⟩ and |Ψ⁻⟩ are constructed by applying H + CNOT to different input states.
   It is verified that they form an orthonormal basis of the 2-qubit space.

2. **Entanglement.** It is demonstrated that no Bell state can be
   written as a tensor product of two independent qubits.
   The von Neumann entropy of the partial trace is computed.

3. **Quantum correlations.** Measurements of the entangled pair are simulated
   and it is verified that the results are perfectly correlated,
   regardless of the measurement order.

4. **Quantum teleportation.** The teleportation protocol of
   Bennett et al. (1993) is implemented: prepare the Bell pair, apply local operations
   at Alice's side, send 2 classical bits, and reconstruct the state at Bob's side.

5. **Proposed exercises.** Verify the Bell basis, demonstrate non-separability,
   implement the superdense coding protocol.

**Modules used:** `src/quantum_gates.py`, `src/quantum_math.py`, `src/visualization.py`

---

### Chapter 3 · Classical Quantum Algorithms

#### `ch03_algoritmos_clasicos/01_deutsch_jozsa.ipynb`

**Chapter objective:** Demonstrate the first exponential quantum advantage
over classical algorithms: the Deutsch-Jozsa problem.

**Notebook structure:**

1. **The problem.** Given a function f: {0,1}ⁿ → {0,1} with the promise that it is
   either constant or balanced, determine which with the fewest number of queries.
   Classical: needs up to 2^(n-1)+1 queries. Quantum: 1 query suffices.

2. **Oracles.** Constant oracles (f = 0 or f = 1)
   and balanced oracles (f(x) = parity of x, or arbitrary XOR functions) are implemented for n qubits.

3. **Quantum circuit.** The Deutsch-Jozsa circuit is built:
   H⊗(n+1) → oracle → H⊗n → measurement.

4. **Verification.** The circuit is run with Qiskit Aer and it is verified that
   measuring |0...0⟩ implies a constant function, and any other result implies balanced.

5. **Proposed exercises.** Implement the oracle for f(x) = x₀ XOR x₁,
   analyze the n=1 case (original Deutsch), estimate the speedup as a function of n.

**Modules used:** `src/quantum_gates.py`, `src/visualization.py`

---

#### `ch03_algoritmos_clasicos/02_bernstein_vazirani.ipynb`

**Objective:** Present the Bernstein-Vazirani algorithm as an extension of
Deutsch-Jozsa: find a hidden bit string s of n bits with a single query.

**Notebook structure:**

1. **The problem.** Given f(x) = s·x mod 2 (binary dot product),
   determine s. A classical algorithm needs n queries (one per bit).
   The quantum algorithm needs 1 query for any n.

2. **Oracle.** The oracle for an arbitrary secret string s is built
   using CNOT: `|x⟩|y⟩ → |x⟩|y ⊕ s·x⟩`.

3. **Circuit.** It is identical to Deutsch-Jozsa: H⊗(n+1) → oracle → H⊗n → measurement.
   Upon measuring, one obtains exactly `|s⟩` with probability 1.

4. **Verification.** Tested for several secret strings and confirmed
   that the measurement always returns s exactly.

5. **Exercises.** What happens if s = 0...0? What if the oracle has noise?

**Modules used:** `src/quantum_gates.py`, `src/visualization.py`

---

#### `ch03_algoritmos_clasicos/03_simon.ipynb`

**Objective:** Present Simon's algorithm, which is the conceptual foundation
of Shor's algorithm and establishes the cornerstone of quantum cryptography.

**Notebook structure:**

1. **The problem.** Given f: {0,1}ⁿ → {0,1}ⁿ with the promise that there exists
   a hidden vector s ≠ 0 such that f(x) = f(y) ⟺ y = x ⊕ s, find s.
   Classical: O(2^(n/2)) queries. Quantum: O(n) queries.

2. **Oracle.** Simon's oracle implementing the periodic function
   is built using crossed CNOTs and XOR with s.

3. **Quantum circuit.** H⊗n → oracle → H⊗n → measurement on the input qubits.
   Each measurement produces a vector y such that y·s = 0 (mod 2).

4. **Classical post-processing.** n-1 linearly independent vectors y are collected
   and the system `Ay = 0 (mod 2)` is solved with
   Gaussian elimination mod 2 to recover s.

5. **Verification.** Tested with s = '1101' and confirmed that the recovered
   period matches the original.

6. **Exercises.** Analyze the failure probability with k measurements,
   implement the case s = 0 (injective function).

**Modules used:** `src/quantum_gates.py`

---

### Chapter 4 · Quantum Fourier Transform

#### `ch04_fourier_cuantica/01_qft.ipynb`

**Chapter objective:** Implement and understand the Quantum Fourier
Transform (QFT), a fundamental building block of Shor, phase estimation, and other algorithms.

**Notebook structure:**

1. **Mathematical definition.** The QFT maps
   `|j⟩ → (1/√N) Σₖ e^(2πijk/N) |k⟩`.
   Compared with the classical DFT, with an explanation of the complexity reduction:
   O(n²) quantum gates vs O(n·2ⁿ) classical operations.

2. **QFT circuit.** The circuit is built with H and controlled phase CR_k gates,
   following the standard decomposition of Nielsen & Chuang.

3. **Numerical verification.** The QFT matrix from Qiskit is computed and
   compared with the analytical matrix from `QuantumMath.qft_matrix(n)`.
   The maximum difference is verified to be < 10⁻¹⁰.

4. **Inverse QFT.** QFT† is built by reversing the order and conjugating
   the phases. QFT⁻¹ · QFT = I is verified.

5. **Exercises.** Implement QFT for n=1, 2, 3, 4 qubits and verify
   unitarity. Compare with `np.fft.fft`. Prove QFT|0...0⟩ = |+...+⟩.

**Modules used:** `src/quantum_gates.py`, `src/quantum_math.py`, `src/visualization.py`

---

### Chapter 5 · Quantum Search

#### `ch05_busqueda/01_grover.ipynb`

**Chapter objective:** Implement Grover's algorithm and demonstrate
the quadratic speedup over classical search.

**Notebook structure:**

1. **The problem.** Search for a marked element in an unstructured database of N
   entries. Classical: O(N). Quantum: O(√N).

2. **The oracle.** The MCZ (Multi-Controlled-Z) oracle that
   flips the phase of the marked state is implemented: `|x⟩ → -|x⟩` if f(x)=1.

3. **The diffusion operator.** The inversion about the mean is implemented:
   `D = 2|s⟩⟨s| - I`, where |s⟩ is the uniform superposition.

4. **Optimal number of iterations.** The optimal number of iterations
   `k = round(π√N/4)` is computed and the probability of the marked state is plotted as a function of k.

5. **Full simulation.** The circuit is run for N=8 (3 qubits) and
   N=16 (4 qubits) with Qiskit Aer. The measurement histogram is shown.

6. **Exercises.** What happens with multiple marked elements? With k too
   large? Implement Grover to find the solution of a 3-SAT.

**Modules used:** `src/quantum_gates.py`, `src/visualization.py`

---

### Chapter 6 · Factorization

#### `ch06_factorizacion/01_shor.ipynb`

**Chapter objective:** Implement Shor's algorithm for N=15,
the smallest pedagogical case that allows seeing the full structure of the algorithm.

**Notebook structure:**

1. **Algorithm structure.** Shor's algorithm consists of two parts:
   (a) classical reduction to order-finding, and (b) quantum phase
   estimation (QPE) to find the period r of f(x) = aˣ mod N.

2. **Quantum Phase Estimation (QPE).** The QPE circuit
   with inverse QFT is built to estimate the phase of the eigenvalues of the operator U|y⟩ = |ay mod N⟩.

3. **Continued fractions.** The classical continued fractions algorithm
   is used to recover the period r from the phase estimate.

4. **Factors.** With r known, the factors are computed as
   `GCD(aʳ/² ± 1, N)`. For N=15, a=2, r=4, the result is 3 and 5.

5. **Full verification.** The complete circuit is run in Qiskit Aer
   and the factorization 15 = 3 × 5 is confirmed to be correct.

6. **Exercises.** Why must r be even and aʳ/² ≢ -1 (mod N)?
   How many qubits are needed to factorize N=21?

**Modules used:** `src/quantum_math.py`, `src/visualization.py`

---

### Chapter 7 · Variational Algorithms

#### `ch07_variacional/01_vqe.ipynb`

**Chapter objective:** Introduce VQE (Variational Quantum Eigensolver)
as a hybrid quantum-classical algorithm for finding the ground state
of a Hamiltonian.

**Notebook structure:**

1. **2-qubit Heisenberg Hamiltonian.** The Hamiltonian
   `H = Jx XX + Jy YY + Jz ZZ + hx (ZI + IZ)` is defined as SparsePauliOp. Its
   exact eigenvalues are computed as a reference.

2. **RealAmplitudes ansatz.** An ansatz of Ry rotation layers and
   CNOT entanglement is used. The reason it is more suitable than UCCSD
   for a spin model is explained.

3. **VQE loop.** VQE is configured with StatevectorEstimator and COBYLA.
   The energy at each evaluation is recorded via a callback.

4. **Convergence curve.** The energy is plotted as a function of iteration
   and compared with the exact value.

5. **Exercises.** Vary Jx, Jy, Jz and analyze how the ground state changes.
   Switch to SPSA to simulate noise. Increase the number of ansatz layers.

**Modules used:** `src/visualization.py`

---

#### `ch07_variacional/02_qaoa.ipynb`

**Objective:** Implement QAOA (Quantum Approximate Optimization Algorithm)
for the MaxCUT problem, the canonical example of quantum combinatorial optimization.

**Notebook structure:**

1. **The MaxCUT problem.** Given a graph G=(V,E), find the bipartition
   (S, S̄) that maximizes the number of edges between S and S̄.

2. **Cost Hamiltonian.** The Hamiltonian
   `H_C = Σ_(ij)∈E w_ij (I - Z_i Z_j) / 2` is built as SparsePauliOp.

3. **QAOA ansatz.** The QAOA circuit of depth p is built with alternating layers
   of the cost operator `e^(-iγH_C)` and the mixing operator
   `e^(-iβH_B)` where `H_B = Σ_i X_i`.

4. **Optimization.** The angles (γ, β) are optimized with COBYLA and the
   convergence curve of the cut value ⟨H_C⟩ is plotted.

5. **Approximation analysis.** The approximation ratio
   `⟨H_C⟩ / MaxCUT_exact` is computed and compared with the theoretical bound of Farhi et al.

6. **Exercises.** Repeat for p=2 and p=3. Does quality improve with p? When
   is the exact optimum reached?

**Modules used:** `src/visualization.py`

---

#### `ch07_variacional/03_vqe_h2.ipynb`

**Objective:** Full demonstration of VQE for the hydrogen molecule H₂,
integrating classical computational chemistry (PySCF) with quantum computing (Qiskit).

**Notebook structure:**

1. **Molecular specification.** Equilibrium geometry of H₂ (R=0.7414 Å),
   STO-3G basis, charge 0, spin 0 (singlet).

2. **PySCF driver.** Computes the one-electron (h_pq) and two-electron
   (h_pqrs) integrals with Hartree-Fock. Returns the Hamiltonian in second
   quantization as FermionicOp.

3. **Jordan-Wigner mapping.** Converts the fermionic Hamiltonian to Pauli
   operators on 4 qubits (2 spatial orbitals × 2 spins). Result: 15 Pauli terms.

4. **UCCSD ansatz.** The initial state is the Hartree-Fock state.
   UCCSD adds unitary single and double excitations. For H₂/STO-3G: 3 parameters.

5. **VQE with SLSQP.** VQE is run and the convergence is plotted.
   The energy converges to FCI with error < 1 mHa (within chemical accuracy).

6. **H₂ dissociation curve.** The total energy is computed for 18 distances
   (0.35–2.50 Å) and HF, exact FCI, and VQE-UCCSD are compared in two panels:
   absolute energies and residual VQE vs FCI error.

7. **Exercises.** Compute with the 6-31G basis. Use COBYLA. Switch to RealAmplitudes.
   Implement noise simulation.

**Modules used:** none from `src/` (uses Qiskit Nature + PySCF directly)

**Requires:** `qiskit-nature`, `pyscf`

---

### Chapter 8 · Quantum Machine Learning

#### `ch08_machine_learning/01_qsvm.ipynb`

**Chapter objective:** Introduce the quantum kernel as a mechanism for
classifying data in high-dimensional Hilbert spaces.

**Notebook structure:**

1. **The quantum kernel.** `K(x, x') = |⟨φ(x)|φ(x')⟩|²` is defined, where
   `|φ(x)⟩ = U(x)|0...0⟩` is the state produced by the feature map.

2. **ZZFeatureMap.** The 2-qubit feature map with
   reps=2 is built: layers of Hadamard gates and ZZ phase gates correlated with the data.

3. **Kernel computation.** `FidelityQuantumKernel` is used to compute the
   kernel matrix between all pairs of training and test points.

4. **SVM with precomputed kernel.** An `SVC(kernel='precomputed')`
   from sklearn is trained with the quantum kernel matrix.

5. **Comparison.** The accuracy of QSVM is compared with a classical RBF SVM
   on the "two moons" dataset (100 points, 25% test).

6. **Exercises.** Try PauliFeatureMap. Increase reps. Apply to the Iris dataset.

**Modules used:** `src/visualization.py`

**Requires:** `qiskit-machine-learning`

---

#### `ch08_machine_learning/02_qnn.ipynb`

**Objective:** Implement a Quantum Neural Network (QNN) and train it for
binary classification, discussing the barren plateau problem.

**Notebook structure:**

1. **QNN architecture.** The circuit combines:
   - **Embedding layer** (ZZFeatureMap): encodes the data x into the quantum state.
   - **Variational layer** (RealAmplitudes): trainable parameters θ.
   - **Output observable**: expected value of Z⊗I as the decision function.

2. **EstimatorQNN.** The QNN is configured with Qiskit by connecting the circuit,
   the input parameters (data), and the weight parameters (trainable).

3. **Training with COBYLA.** `NeuralNetworkClassifier` is used with a
   callback that records the loss at each iteration.

4. **Learning curve.** The loss is plotted as a function of iteration and
   the accuracy on the test set is evaluated.

5. **Barren plateau.** The reason why the gradient variance scales
   exponentially with the number of qubits and layers is discussed, along with strategies
   to mitigate it (smart initialization, structured ansatz).

6. **Exercises.** Switch to ADAM with parameter-shift. Increase ansatz layers.
   Compare with an equivalent classical neural network.

**Modules used:** `src/visualization.py`

**Requires:** `qiskit-machine-learning`

---

## How to add a new chapter

1. Create the directory `notebooks/ch09_<topic>/`.
2. Create the notebook following the template:
   - Markdown cell: **Objectives** (3-5 concrete points).
   - Code cell: imports with `sys.path.insert`.
   - Alternate Markdown cells (theory with LaTeX) and code cells (implementation).
   - Final Markdown cell: **Proposed exercises** (at least 3).
3. If the new topic requires new mathematical utilities, add them to `src/`
   with NumPy-style docstrings and their corresponding tests in `tests/`.
4. Update `README.md` and `GUIA.md` with the new entry.

---

## Roadmap

| Priority | Proposed extension | Suggested folder |
|---|---|---|
| High | Quantum error correction (Steane, Shor 9-qubit) | `ch09_error_correction/` |
| High | Noise simulation (Kraus channels, Lindblad) | `src/noise_models.py` |
| Medium | Quantum phase estimation (standalone QPE) | `ch04_fourier_cuantica/02_qpe.ipynb` |
| Medium | VQLS (Variational Quantum Linear Solver) | `ch07_variacional/04_vqls.ipynb` |
| Low | Execution on real IBM Quantum hardware | `ch10_hardware/` |
| Low | Alternative backends (PennyLane, Cirq) | `src/backends/` |
