# Guía de Uso del Repositorio

Bienvenido al repositorio de computación cuántica que acompaña al libro.
Esta guía explica para qué sirve cada fichero y cómo trabajar con él.

---

## Estructura general

```
./
├── GUIA.md                    ← Este fichero
├── README.md                  ← Portada del repositorio (instalación, requisitos)
├── requirements.txt           ← Dependencias pip
├── environment.yml            ← Entorno Conda completo
├── setup.py                   ← Instalación del paquete src/ como módulo
│
├── src/                       ← Módulos reutilizables (importados desde los notebooks)
├── notebooks/                 ← Notebooks pedagógicos organizados por capítulo
├── docs/                      ← Documentación técnica
└── tests/                     ← Tests automáticos de los módulos src/
```

---

## 1. Cómo empezar

### Instalación del entorno (una sola vez)

**Opción A — Conda (recomendado):**

```bash
conda env create -f environment.yml
conda activate quantum-book
```

**Opción B — pip:**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

### Lanzar JupyterLab

```bash
jupyter lab
```

---

## 2. Notebooks — guía por capítulo

Todos los notebooks están en `notebooks/` organizados por capítulo.
Ábrelos desde JupyterLab y ejecuta las celdas en orden.

---

### Capítulo 1 · Introducción a los Qubits (`ch01_introduccion/`)

#### `01_qubits_y_estados.ipynb`

**¿Qué hace?**
Introduce la representación matemática de un qubit, las puertas de un qubit
y la visualización en la esfera de Bloch. Es el punto de entrada recomendado
para cualquier lector que empieza el libro.

**Contenido:**
- Estado de un qubit: amplitudes α y β, probabilidades.
- Puertas H, X, Y, Z, S, T como matrices unitarias.
- Cálculo del vector de Bloch.
- Visualización estática con matplotlib.

**Módulos que usa:**
`src/quantum_gates.py`, `src/quantum_math.py`, `src/visualization.py`

---

#### `02_bloch_interactivo.ipynb`

**¿Qué hace?**
Simulador interactivo de la esfera de Bloch con controles `ipywidgets`
directamente dentro de JupyterLab. Sin necesidad de ninguna aplicación externa.

**Cómo usarlo:**
1. Ejecuta `Kernel → Restart & Run All`.
2. Aparece un panel con controles a la izquierda y la esfera 3D a la derecha.
3. Elige el estado inicial, aplica puertas fijas o paramétricas con sliders,
   o ejecuta secuencias predefinidas.
4. La celda final permite escribir tu propia secuencia de puertas.

**Módulos que usa:**
`src/bloch_simulator.py`, `src/quantum_math.py`

> **Requisito adicional:** `ipywidgets`
> ```bash
> pip install ipywidgets
> ```

---

### Capítulo 2 · Entrelazamiento (`ch02_entrelazamiento/`)

#### `01_bell_states.ipynb`

**¿Qué hace?**
Construye y analiza los cuatro estados de Bell. Introduce el entrelazamiento
cuántico, la entropía de Von Neumann y la teleportación cuántica.

**Contenido:**
- Creación de estados de Bell con H + CNOT.
- Medida de la entropía de entrelazamiento.
- Protocolo de teleportación cuántica (simulación sin ruido).

---

### Capítulo 3 · Algoritmos Clásicos (`ch03_algoritmos_clasicos/`)

#### `01_deutsch_jozsa.ipynb`

**¿Qué hace?**
Implementa el algoritmo de Deutsch-Jozsa: el primer ejemplo de ventaja cuántica
exponencial sobre un algoritmo clásico determinista.

**Contenido:**
- Oráculos constantes y balanceados para n bits.
- Demostración de que 1 consulta cuántica basta frente a 2^(n-1)+1 clásicas.
- Visualización del circuito y de las medidas.

---

#### `02_bernstein_vazirani.ipynb`

**¿Qué hace?**
Resuelve el problema Bernstein-Vazirani: encontrar una cadena oculta de $n$ bits
con una sola consulta al oráculo.

**Contenido:**
- Construcción del oráculo para una cadena secreta arbitraria.
- Verificación de que el circuito recupera la cadena exacta en una sola medida.

---

#### `03_simon.ipynb`

**¿Qué hace?**
Implementa el algoritmo de Simon para encontrar el período oculto de una función.
Es la base conceptual del algoritmo de Shor.

**Contenido:**
- Oráculo de Simon, circuito cuántico (aplicación doble de Hadamard).
- Post-procesamiento clásico: álgebra lineal mod 2.
- Verificación de que `y · s = 0 (mod 2)` para cada medida.

---

### Capítulo 4 · Transformada de Fourier Cuántica (`ch04_fourier_cuantica/`)

#### `01_qft.ipynb`

**¿Qué hace?**
Implementa la Transformada de Fourier Cuántica (QFT) y verifica su corrección
comparando con la matriz analítica obtenida de forma exacta.

**Contenido:**
- Definición de la QFT y su circuito (puertas H y CR_k controladas de fase).
- Verificación numérica: `QFT_Qiskit ≈ QFT_analítica`.
- Análisis de la complejidad: O(n²) puertas vs O(n·2ⁿ) de la FFT clásica.

---

### Capítulo 5 · Búsqueda Cuántica (`ch05_busqueda/`)

#### `01_grover.ipynb`

**¿Qué hace?**
Implementa el algoritmo de Grover para búsqueda de un elemento en una base de datos.
Ganancia cuadrática sobre la búsqueda clásica: O(√N) vs O(N).

**Contenido:**
- Oráculo de Grover Multi-Controlled-Z (MCZ).
- Operador de difusión de Grover (inversión sobre la media).
- Número óptimo de iteraciones: π√N/4.
- Gráfico de probabilidad del ítem marcado en función de las iteraciones.

---

### Capítulo 6 · Factorización (`ch06_factorizacion/`)

#### `01_shor.ipynb`

**¿Qué hace?**
Implementa el algoritmo de Shor para N = 15 (caso pedagógico).
Combina estimación de fase cuántica (QPE) con el algoritmo clásico de fracciones continuas.

**Contenido:**
- Estructura del algoritmo: QPE + reducción clásica al MCD.
- Circuito completo para N = 15, a = 2 (período r = 4).
- Verificación de los factores: 15 = 3 × 5.

---

### Capítulo 7 · Algoritmos Variacionales (`ch07_variacional/`)

#### `01_vqe.ipynb`

**¿Qué hace?**
Implementa el VQE (Variational Quantum Eigensolver) para encontrar la energía
del estado fundamental del Hamiltoniano de Heisenberg de 2 qubits.

**Contenido:**
- Definición del Hamiltoniano de Heisenberg como SparsePauliOp.
- Ansatz RealAmplitudes, optimizador COBYLA.
- Curva de convergencia de la energía.

---

#### `02_qaoa.ipynb`

**¿Qué hace?**
Implementa QAOA (Quantum Approximate Optimization Algorithm) para el problema
MaxCUT en un grafo de 4 nodos.

**Contenido:**
- Hamiltoniano de coste para MaxCUT: H_C = Σ w_ij (I − Z_i Z_j) / 2.
- Ansatz QAOA de profundidad p con capas de coste y mezcla.
- Optimización con COBYLA y análisis del ratio de aproximación.

---

#### `03_vqe_h2.ipynb`

**¿Qué hace?**
Demostración completa de VQE para la molécula de hidrógeno H₂, basada en
cálculos reales con PySCF y Qiskit Nature.

**Contenido:**
1. Driver PySCF → Hamiltoniano fermiónico de segunda cuantización.
2. Mapeo Jordan-Wigner → Hamiltoniano de 4 qubits (15 términos de Pauli).
3. Ansatz UCCSD con estado inicial Hartree-Fock (3 parámetros).
4. VQE con SLSQP: convergencia al FCI con error < 1 mHa.
5. Curva de disociación H₂: energías HF, FCI y VQE para R = 0.35–2.50 Å.

> **Requiere:** `qiskit-nature`, `pyscf`
> ```bash
> pip install qiskit-nature pyscf
> ```

---

### Capítulo 8 · Machine Learning Cuántico (`ch08_machine_learning/`)

#### `01_qsvm.ipynb`

**¿Qué hace?**
Implementa una Máquina de Vectores de Soporte con kernel cuántico (QSVM)
y la compara con un SVM clásico de kernel RBF.

**Contenido:**
- Mapa de características ZZFeatureMap (2 qubits, reps=2).
- Kernel cuántico: K(x, x') = |⟨φ(x)|φ(x')⟩|².
- Clasificación del dataset "dos lunas" con SVM precomputado.
- Comparación de exactitud cuántico vs clásico.

> **Requiere:** `qiskit-machine-learning`
> ```bash
> pip install qiskit-machine-learning
> ```

---

#### `02_qnn.ipynb`

**¿Qué hace?**
Implementa una Red Neuronal Cuántica (QNN) con `EstimatorQNN` de Qiskit
para clasificación binaria de círculos concéntricos.

**Contenido:**
- Arquitectura: ZZFeatureMap (embedding) + RealAmplitudes (variacional).
- Entrenamiento con COBYLA y `NeuralNetworkClassifier`.
- Curva de aprendizaje y exactitud final.
- Discusión del problema del barren plateau.

---

## 3. Módulos `src/` — referencia técnica

Los módulos de `src/` son la biblioteca compartida del repositorio.
**No son scripts ejecutables**; se importan desde los notebooks con:

```python
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '..', '..'))
from src.quantum_gates import Gates
```

### `src/quantum_gates.py`

Clase `Gates` con todas las puertas cuánticas como matrices NumPy `complex128`.

| Método / Atributo | Descripción |
|---|---|
| `Gates.I`, `Gates.X`, `Gates.Y`, `Gates.Z` | Puertas de Pauli |
| `Gates.H` | Hadamard |
| `Gates.S`, `Gates.T`, `Gates.Sdg`, `Gates.Tdg` | Puertas de fase |
| `Gates.CNOT`, `Gates.CZ`, `Gates.SWAP` | Puertas de 2 qubits |
| `Gates.Rx(θ)`, `Gates.Ry(θ)`, `Gates.Rz(θ)` | Rotaciones paramétricas |
| `Gates.P(φ)` | Puerta de fase |
| `Gates.CR_k(k)` | Rotación controlada de fase (para QFT) |
| `Gates.tensor(*gates)` | Producto tensorial de varias puertas |
| `Gates.is_unitary(U)` | Verifica U†U = I |
| `Gates.apply(U, state)` | Aplica U al estado |

---

### `src/quantum_math.py`

Clase `QuantumMath` con herramientas de álgebra cuántica.

| Método | Descripción |
|---|---|
| `QuantumMath.ket0()`, `QuantumMath.ket1()` | Estados base |
| `QuantumMath.ket_plus()`, `QuantumMath.ket_minus()` | Estados de Hadamard |
| `QuantumMath.tensor_product(s1, s2)` | Producto tensorial de estados |
| `QuantumMath.probabilities(state)` | Vector de probabilidades de medida |
| `QuantumMath.inner_product(s1, s2)` | Producto interno ⟨s1|s2⟩ |
| `QuantumMath.bloch_vector(state)` | Vector de Bloch (x, y, z) |
| `QuantumMath.bloch_angles(state)` | Ángulos polares (θ, φ) |
| `QuantumMath.density_matrix(state)` | Matriz densidad ρ = |ψ⟩⟨ψ| |
| `QuantumMath.fidelity(s1, s2)` | Fidelidad F = \|⟨s1\|s2⟩\|² |
| `QuantumMath.von_neumann_entropy(rho)` | Entropía S = -Tr(ρ log ρ) |
| `QuantumMath.qft_matrix(n)` | Matriz QFT analítica para n qubits |
| `QuantumMath.normalize(state)` | Normaliza un vector de estado |

---

### `src/visualization.py`

Clase `QuantumVisualization` con gráficos en estilo oscuro académico.

| Método | Descripción |
|---|---|
| `plot_histogram(counts)` | Histograma de medidas (resultados de shots) |
| `plot_bloch_vector(state)` | Esfera de Bloch estática (matplotlib) |
| `plot_unitary(U, title)` | Mapa de calor de una matriz unitaria |
| `plot_statevector(state)` | Amplitudes y probabilidades del statevector |

---

### `src/bloch_simulator.py`

Clase `BlochSimulator` para simulación interactiva de un qubit.
Es el motor del notebook `02_bloch_interactivo.ipynb`.

| Método | Descripción |
|---|---|
| `BlochSimulator(initial_state)` | Crea el simulador con estado inicial |
| `apply_gate(gate, **kwargs)` | Aplica una puerta y registra la posición |
| `apply_sequence([{gate,...}])` | Aplica una lista de puertas |
| `plot_trajectory(title, dark_mode)` | Figura Plotly 3D con la trayectoria |
| `state_info()` | Devuelve dict con α, β, probabilidades, Bloch |
| `reset(initial_state)` | Reinicia el simulador |

---

## 4. Tests

Los tests verifican matemáticamente los módulos `src/`.

```bash
# Ejecutar desde la raíz del repositorio
pytest tests/ -v
```

| Fichero | Qué verifica |
|---|---|
| `tests/test_gates.py` | Unitariedad de puertas, identidades algebraicas (HXH=Z, T⁸=I, ...) |
| `tests/test_math.py` | Normalización, vector de Bloch, fidelidad, entropía, QFT unitaria |

---

## 5. Convenciones del código

- **Python**: PEP-8 estricto. Funciones con docstrings NumPy-style.
- **Notebooks**: una celda = una idea. Sin celdas de más de 40 líneas.
- **LaTeX**: siempre en `$$...$$` (display) o `$...$` (inline).
- **Gráficos**: usar `QuantumVisualization` de `src/` para mantener el estilo unificado.
- **Seeds**: siempre `np.random.seed(42)` al inicio de notebooks con aleatoriedad.
- **Paths**: siempre relativos a la carpeta del fichero (nunca rutas absolutas de disco).
