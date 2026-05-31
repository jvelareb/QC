# Arquitectura del Repositorio

Este documento describe la estructura completa del repositorio, el papel de cada
carpeta y el contenido detallado de cada notebook Jupyter.

---

## Árbol de carpetas

```
./
├── GUIA.md                  ← Guía de uso rápido (cómo instalar, cómo ejecutar)
├── README.md                ← Portada del repositorio
├── requirements.txt         ← Dependencias pip
├── environment.yml          ← Entorno Conda reproducible
├── setup.py                 ← Instalación del paquete src/ como módulo Python
│
├── src/                     ← Biblioteca de código reutilizable
│   ├── quantum_gates.py     ← Todas las puertas cuánticas como matrices NumPy
│   ├── quantum_math.py      ← Álgebra cuántica (Bloch, fidelidad, QFT, entropía)
│   ├── bloch_simulator.py   ← Simulador interactivo de 1 qubit (Plotly 3D)
│   └── visualization.py     ← Gráficos unificados con estilo oscuro académico
│
├── notebooks/               ← Notebooks pedagógicos por capítulo
│   ├── ch01_introduccion/
│   ├── ch02_entrelazamiento/
│   ├── ch03_algoritmos_clasicos/
│   ├── ch04_fourier_cuantica/
│   ├── ch05_busqueda/
│   ├── ch06_factorizacion/
│   ├── ch07_variacional/
│   └── ch08_machine_learning/
│
├── docs/                    ← Documentación técnica
│   ├── arquitectura.md      ← Este fichero
│   └── guia_instalacion.md  ← Instrucciones de instalación detalladas
│
└── tests/                   ← Tests automáticos de los módulos src/
    ├── test_gates.py
    └── test_math.py
```

---

## Principio de diseño

**Separación de responsabilidades.** El código matemático vive en `src/` y los
notebooks sólo lo invocan. Si se mejora una función en `quantum_math.py`, el
cambio se propaga automáticamente a todos los notebooks sin tocar ninguno.

**Progresión didáctica.** Los capítulos van de lo más simple (qubit, puerta única)
a lo más complejo (VQE para H₂, redes neuronales cuánticas), siguiendo el orden
del libro.

**Paths siempre relativos.** Ningún fichero contiene rutas absolutas de disco.
Todos los imports usan `os.path.join(os.getcwd(), '..', '..')` para localizar `src/`.

---

## Módulos `src/` — resumen

| Módulo | Clase principal | Para qué sirve |
|---|---|---|
| `quantum_gates.py` | `Gates` | Puertas cuánticas como matrices NumPy: H, X, Y, Z, S, T, CNOT, CZ, Rx, Ry, Rz, P, CR_k... |
| `quantum_math.py` | `QuantumMath` | Estados básicos, probabilidades, vector de Bloch, fidelidad, entropía de Von Neumann, QFT analítica |
| `bloch_simulator.py` | `BlochSimulator` | Aplica secuencias de puertas a un qubit, registra la trayectoria y genera la figura Plotly 3D |
| `visualization.py` | `QuantumVisualization` | Histogramas de medidas, esfera de Bloch estática, mapas de calor de unitarias, statevectors |

---

## Notebooks — descripción detallada

---

### Capítulo 1 · Introducción a los Qubits

#### `ch01_introduccion/01_qubits_y_estados.ipynb`

**Objetivo del capítulo:** Presentar las ideas fundamentales de la computación
cuántica: el qubit como vector en ℂ², las puertas cuánticas como matrices unitarias,
y la esfera de Bloch como herramienta de visualización.

**Estructura del notebook:**

1. **Representación matemática del qubit.** Un qubit es un vector
   `|ψ⟩ = α|0⟩ + β|1⟩` con `|α|² + |β|² = 1`. Se calcula el vector de Bloch
   (x, y, z) a partir de las amplitudes y se visualiza en la esfera.

2. **Puertas de un qubit.** Se construyen las puertas H, X, Y, Z, S, T
   como matrices 2×2 y se verifica su unitariedad con `Gates.is_unitary()`.

3. **Aplicación de puertas.** Se aplican secuencias de puertas al estado |0⟩
   y se representan los estados intermedios en la esfera de Bloch.

4. **Probabilidades de medida.** Se calcula `P(|0⟩) = |α|²` y
   `P(|1⟩) = |β|²` y se simula una medida con `np.random.choice`.

5. **Ejercicios propuestos.** Verificar HXH = Z, calcular la acción de T⁸,
   encontrar el estado que maximiza P(|1⟩) tras aplicar Rx(θ).

**Módulos usados:** `src/quantum_gates.py`, `src/quantum_math.py`, `src/visualization.py`

---

#### `ch01_introduccion/02_bloch_interactivo.ipynb`

**Objetivo:** Explorar la esfera de Bloch de forma totalmente interactiva
dentro de JupyterLab, sin ninguna aplicación externa.

**Estructura del notebook:**

1. **Panel de controles (ipywidgets).** Se construye una interfaz con:
   - Dropdown de estado inicial: |0⟩, |1⟩, |+⟩, |-⟩, |i⟩, |-i⟩.
   - Botón "Reiniciar" que resetea el simulador al estado seleccionado.
   - Dropdown de puerta fija (H, X, Y, Z, S, T, Sdg, Tdg) + botón "Aplicar".
   - Dropdown de puerta paramétrica (Rx, Ry, Rz, P) + slider de ángulo en grados.
   - Dropdown de secuencias predefinidas (Hadamard, teleportación, T⁸ = I...).
   - Botón "Limpiar trayectoria" que borra el rastro sin cambiar el estado.

2. **Esfera de Bloch 3D (Plotly).** Se actualiza en tiempo real tras cada acción.
   La trayectoria se muestra como una línea roja con marcadores.
   Al pasar el ratón sobre un punto aparece el nombre de la puerta y las coordenadas.

3. **Panel de información.** Muestra en todo momento:
   amplitudes α y β, probabilidades P(|0⟩) y P(|1⟩),
   vector de Bloch (x, y, z) e historial de puertas aplicadas.

4. **Sección educativa.** Tabla de estados básicos, tabla de puertas con su efecto
   en la esfera de Bloch y las fórmulas matemáticas del vector de Bloch.

5. **Experimento libre.** Celda de código donde el alumno puede escribir
   su propia secuencia con `apply_sequence([...])` y ver el resultado.

**Módulos usados:** `src/bloch_simulator.py`, `src/quantum_math.py`

**Requiere:** `ipywidgets` (`pip install ipywidgets`)

---

### Capítulo 2 · Entrelazamiento

#### `ch02_entrelazamiento/01_bell_states.ipynb`

**Objetivo del capítulo:** Introducir el entrelazamiento cuántico como la
propiedad más característica de la computación cuántica, sin analogía clásica.

**Estructura del notebook:**

1. **Los cuatro estados de Bell.** Se construyen los estados
   |Φ⁺⟩, |Φ⁻⟩, |Ψ⁺⟩ y |Ψ⁻⟩ aplicando H + CNOT sobre distintos estados de entrada.
   Se verifica que forman una base ortonormal del espacio de 2 qubits.

2. **Entrelazamiento.** Se demuestra que ningún estado de Bell se puede
   escribir como producto tensorial de dos qubits independientes.
   Se calcula la entropía de von Neumann de la traza parcial.

3. **Correlaciones cuánticas.** Se simulan medidas del par entrelazado
   y se comprueba que los resultados están perfectamente correlacionados,
   independientemente del orden de medida.

4. **Teleportación cuántica.** Se implementa el protocolo de teleportación
   de Bennett et al. (1993): preparar el par de Bell, aplicar operaciones locales
   en Alice, enviar 2 bits clásicos, y reconstruir el estado en Bob.

5. **Ejercicios propuestos.** Verificar la base de Bell, demostrar la no-separabilidad,
   implementar el protocolo de codificación superdensa.

**Módulos usados:** `src/quantum_gates.py`, `src/quantum_math.py`, `src/visualization.py`

---

### Capítulo 3 · Algoritmos Cuánticos Clásicos

#### `ch03_algoritmos_clasicos/01_deutsch_jozsa.ipynb`

**Objetivo del capítulo:** Demostrar la primera ventaja cuántica exponencial
sobre algoritmos clásicos: el problema Deutsch-Jozsa.

**Estructura del notebook:**

1. **El problema.** Dada una función f: {0,1}ⁿ → {0,1} con la promesa de que es
   constante o balanceada, determinar cuál es con el menor número de consultas.
   Clásico: necesita hasta 2^(n-1)+1 consultas. Cuántico: 1 consulta basta.

2. **Oráculos.** Se implementan los oráculos constantes (f = 0 ó f = 1)
   y balanceados (f(x) = paridad de x, o funciones XOR arbitrarias) para n qubits.

3. **Circuito cuántico.** Se construye el circuito Deutsch-Jozsa:
   H⊗(n+1) → oráculo → H⊗n → medida.

4. **Verificación.** Se ejecuta el circuito con Qiskit Aer y se comprueba
   que medir |0...0⟩ implica función constante y cualquier otro resultado implica balanceada.

5. **Ejercicios propuestos.** Implementar el oráculo para f(x) = x₀ XOR x₁,
   analizar el caso n=1 (Deutsch original), estimar el speedup en función de n.

**Módulos usados:** `src/quantum_gates.py`, `src/visualization.py`

---

#### `ch03_algoritmos_clasicos/02_bernstein_vazirani.ipynb`

**Objetivo:** Presentar el algoritmo de Bernstein-Vazirani como extensión de
Deutsch-Jozsa: encontrar una cadena oculta s de n bits con una sola consulta.

**Estructura del notebook:**

1. **El problema.** Dada f(x) = s·x mod 2 (producto escalar binario),
   determinar s. Un algoritmo clásico necesita n consultas (una por bit).
   El algoritmo cuántico necesita 1 consulta para cualquier n.

2. **Oráculo.** Se construye el oráculo para una cadena secreta s arbitraria
   usando CNOT: `|x⟩|y⟩ → |x⟩|y ⊕ s·x⟩`.

3. **Circuito.** Es idéntico al de Deutsch-Jozsa: H⊗(n+1) → oráculo → H⊗n → medida.
   Al medir, se obtiene exactamente `|s⟩` con probabilidad 1.

4. **Verificación.** Se prueba para varias cadenas secretas y se confirma
   que la medida siempre devuelve s exactamente.

5. **Ejercicios.** ¿Qué ocurre si s = 0...0? ¿Qué pasa si el oráculo tiene ruido?

**Módulos usados:** `src/quantum_gates.py`, `src/visualization.py`

---

#### `ch03_algoritmos_clasicos/03_simon.ipynb`

**Objetivo:** Presentar el algoritmo de Simon, que es la base conceptual
del algoritmo de Shor y establece la piedra angular de la criptografía cuántica.

**Estructura del notebook:**

1. **El problema.** Dada f: {0,1}ⁿ → {0,1}ⁿ con la promesa de que existe
   un vector oculto s ≠ 0 tal que f(x) = f(y) ⟺ y = x ⊕ s, encontrar s.
   Clásico: O(2^(n/2)) consultas. Cuántico: O(n) consultas.

2. **Oráculo.** Se construye el oráculo de Simon que implementa la función
   periódica usando CNOTs cruzados y XOR con s.

3. **Circuito cuántico.** H⊗n → oráculo → H⊗n → medida en los qubits de entrada.
   Cada medida produce un vector y tal que y·s = 0 (mod 2).

4. **Post-procesamiento clásico.** Se recolectan n-1 vectores y linealmente
   independientes y se resuelve el sistema `Ay = 0 (mod 2)` con eliminación
   gaussiana mod 2 para recuperar s.

5. **Verificación.** Se prueba con s = '1101' y se confirma que el período
   recuperado coincide con el original.

6. **Ejercicios.** Analizar la probabilidad de fallo con k medidas,
   implementar el caso s = 0 (función inyectiva).

**Módulos usados:** `src/quantum_gates.py`

---

### Capítulo 4 · Transformada de Fourier Cuántica

#### `ch04_fourier_cuantica/01_qft.ipynb`

**Objetivo del capítulo:** Implementar y comprender la Transformada de Fourier
Cuántica (QFT), bloque fundamental de Shor, estimación de fase y otros algoritmos.

**Estructura del notebook:**

1. **Definición matemática.** La QFT mapea
   `|j⟩ → (1/√N) Σₖ e^(2πijk/N) |k⟩`.
   Se compara con la DFT clásica y se explica la reducción de complejidad:
   O(n²) puertas cuánticas vs O(n·2ⁿ) operaciones clásicas.

2. **Circuito QFT.** Se construye el circuito con puertas H y CR_k controladas
   de fase, siguiendo la descomposición estándar de Nielsen & Chuang.

3. **Verificación numérica.** Se calcula la matriz de la QFT de Qiskit y se
   compara con la matriz analítica de `QuantumMath.qft_matrix(n)`.
   Se verifica que la diferencia máxima es < 10⁻¹⁰.

4. **Inversa de la QFT.** Se construye QFT† invirtiendo el orden y conjugando
   las fases. Se verifica QFT⁻¹ · QFT = I.

5. **Ejercicios.** Implementar QFT para n=1, 2, 3, 4 qubits y verificar la
   unitariedad. Comparar con `np.fft.fft`. Demostrar QFT|0...0⟩ = |+...+⟩.

**Módulos usados:** `src/quantum_gates.py`, `src/quantum_math.py`, `src/visualization.py`

---

### Capítulo 5 · Búsqueda Cuántica

#### `ch05_busqueda/01_grover.ipynb`

**Objetivo del capítulo:** Implementar el algoritmo de Grover y demostrar
la ganancia cuadrática sobre la búsqueda clásica.

**Estructura del notebook:**

1. **El problema.** Buscar un elemento marcado en una base de datos de N
   entradas sin estructura. Clásico: O(N). Cuántico: O(√N).

2. **El oráculo.** Se implementa el oráculo MCZ (Multi-Controlled-Z) que
   invierte la fase del estado marcado: `|x⟩ → -|x⟩` si f(x)=1.

3. **El operador de difusión.** Se implementa la inversión sobre la media:
   `D = 2|s⟩⟨s| - I`, donde |s⟩ es la superposición uniforme.

4. **Número óptimo de iteraciones.** Se calcula el número óptimo de iteraciones
   `k = round(π√N/4)` y se traza la probabilidad del estado marcado en función de k.

5. **Simulación completa.** Se ejecuta el circuito para N=8 (3 qubits) y
   N=16 (4 qubits) con Qiskit Aer. Se muestra el histograma de medidas.

6. **Ejercicios.** ¿Qué pasa con múltiples elementos marcados? ¿Con k demasiado
   grande? Implementar Grover para buscar la solución de un 3-SAT.

**Módulos usados:** `src/quantum_gates.py`, `src/visualization.py`

---

### Capítulo 6 · Factorización

#### `ch06_factorizacion/01_shor.ipynb`

**Objetivo del capítulo:** Implementar el algoritmo de Shor para N=15,
el caso pedagógico más pequeño que permite ver toda la estructura del algoritmo.

**Estructura del notebook:**

1. **Estructura del algoritmo.** El algoritmo de Shor consta de dos partes:
   (a) reducción clásica a la búsqueda del orden, y (b) estimación de fase
   cuántica (QPE) para encontrar el período r de f(x) = aˣ mod N.

2. **Estimación de fase cuántica (QPE).** Se construye el circuito QPE
   con QFT inversa para estimar la fase de los autovalores del operador U|y⟩ = |ay mod N⟩.

3. **Fracciones continuas.** Se usa el algoritmo clásico de fracciones continuas
   para recuperar el período r a partir de la estimación de fase.

4. **Factores.** Con r conocido, se calculan los factores como
   `MCD(aʳ/² ± 1, N)`. Para N=15, a=2, r=4, se obtienen 3 y 5.

5. **Verificación completa.** Se ejecuta el circuito completo en Qiskit Aer
   y se confirma que la factorización 15 = 3 × 5 es correcta.

6. **Ejercicios.** ¿Por qué es necesario r par y aʳ/² ≢ -1 (mod N)?
   ¿Cuántos qubits se necesitan para factorizar N=21?

**Módulos usados:** `src/quantum_math.py`, `src/visualization.py`

---

### Capítulo 7 · Algoritmos Variacionales

#### `ch07_variacional/01_vqe.ipynb`

**Objetivo del capítulo:** Introducir el VQE (Variational Quantum Eigensolver)
como algoritmo híbrido cuántico-clásico para encontrar el estado fundamental
de un Hamiltoniano.

**Estructura del notebook:**

1. **Hamiltoniano de Heisenberg de 2 qubits.** Se define el Hamiltoniano
   `H = Jx XX + Jy YY + Jz ZZ + hx (ZI + IZ)` como SparsePauliOp. Se calculan
   sus autovalores exactos como referencia.

2. **Ansatz RealAmplitudes.** Se usa un ansatz de capas de rotaciones Ry y
   entrelazamiento CNOT. Se explica por qué es más adecuado que el UCCSD
   para un modelo de espines.

3. **Bucle VQE.** Se configura el VQE con StatevectorEstimator y COBYLA.
   Se registra la energía en cada evaluación mediante un callback.

4. **Curva de convergencia.** Se traza la energía en función de la iteración
   y se compara con el valor exacto.

5. **Ejercicios.** Variar Jx, Jy, Jz y analizar cómo cambia el estado fundamental.
   Cambiar a SPSA para simular ruido. Aumentar el número de capas del ansatz.

**Módulos usados:** `src/visualization.py`

---

#### `ch07_variacional/02_qaoa.ipynb`

**Objetivo:** Implementar QAOA (Quantum Approximate Optimization Algorithm)
para el problema MaxCUT, el ejemplo canónico de optimización combinatoria cuántica.

**Estructura del notebook:**

1. **El problema MaxCUT.** Dado un grafo G=(V,E), encontrar la bipartición
   (S, S̄) que maximice el número de aristas entre S y S̄.

2. **Hamiltoniano de coste.** Se construye el Hamiltoniano
   `H_C = Σ_(ij)∈E w_ij (I - Z_i Z_j) / 2` como SparsePauliOp.

3. **Ansatz QAOA.** Se construye el circuito QAOA de profundidad p con capas
   alternadas del operador de coste `e^(-iγH_C)` y el operador de mezcla
   `e^(-iβH_B)` donde `H_B = Σ_i X_i`.

4. **Optimización.** Se optimizan los ángulos (γ, β) con COBYLA y se traza
   la curva de convergencia del valor de corte ⟨H_C⟩.

5. **Análisis de aproximación.** Se calcula el ratio de aproximación
   `⟨H_C⟩ / MaxCUT_exacto` y se compara con el bound teórico de Farhi et al.

6. **Ejercicios.** Repetir para p=2 y p=3. ¿Mejora la calidad con p? ¿Cuándo
   se alcanza el óptimo exacto?

**Módulos usados:** `src/visualization.py`

---

#### `ch07_variacional/03_vqe_h2.ipynb`

**Objetivo:** Demostración completa de VQE para la molécula de hidrógeno H₂,
integrando química computacional clásica (PySCF) con computación cuántica (Qiskit).

**Estructura del notebook:**

1. **Especificación molecular.** Geometría de equilibrio del H₂ (R=0.7414 Å),
   base STO-3G, carga 0, spin 0 (singlete).

2. **Driver PySCF.** Calcula los integrales de un electrón (h_pq) y dos
   electrones (h_pqrs) con Hartree-Fock. Devuelve el Hamiltoniano en segunda
   cuantización como FermionicOp.

3. **Mapeo Jordan-Wigner.** Convierte el Hamiltoniano fermiónico a operadores
   de Pauli en 4 qubits (2 orbitales espaciales × 2 espines). Resultado: 15 términos de Pauli.

4. **Ansatz UCCSD.** El estado inicial es el estado Hartree-Fock.
   UCCSD añade excitaciones simples y dobles unitarias. Para H₂/STO-3G: 3 parámetros.

5. **VQE con SLSQP.** Se ejecuta el VQE y se traza la convergencia.
   La energía converge al FCI con error < 1 mHa (dentro de la precisión química).

6. **Curva de disociación H₂.** Se calcula la energía total para 18 distancias
   (0.35–2.50 Å) y se compara HF, FCI exacta y VQE-UCCSD en dos paneles:
   energías absolutas y error residual VQE vs FCI.

7. **Ejercicios.** Calcular con la base 6-31G. Usar COBYLA. Cambiar a RealAmplitudes.
   Implementar simulación con ruido.

**Módulos usados:** ninguno de `src/` (usa directamente Qiskit Nature + PySCF)

**Requiere:** `qiskit-nature`, `pyscf`

---

### Capítulo 8 · Machine Learning Cuántico

#### `ch08_machine_learning/01_qsvm.ipynb`

**Objetivo del capítulo:** Introducir el kernel cuántico como mecanismo para
clasificar datos en espacios de Hilbert de alta dimensión.

**Estructura del notebook:**

1. **El kernel cuántico.** Se define `K(x, x') = |⟨φ(x)|φ(x')⟩|²` donde
   `|φ(x)⟩ = U(x)|0...0⟩` es el estado producido por el mapa de características.

2. **ZZFeatureMap.** Se construye el mapa de características de 2 qubits con
   reps=2: capas de Hadamard y puertas de fase ZZ correlacionadas con los datos.

3. **Cálculo del kernel.** Se usa `FidelityQuantumKernel` para calcular la
   matriz de kernel entre todos los pares de puntos de entrenamiento y test.

4. **SVM con kernel precomputado.** Se entrena un `SVC(kernel='precomputed')`
   de sklearn con la matriz de kernel cuántica.

5. **Comparación.** Se compara la exactitud del QSVM con un SVM clásico RBF
   en el dataset "dos lunas" (100 puntos, 25% test).

6. **Ejercicios.** Probar PauliFeatureMap. Aumentar reps. Aplicar al dataset Iris.

**Módulos usados:** `src/visualization.py`

**Requiere:** `qiskit-machine-learning`

---

#### `ch08_machine_learning/02_qnn.ipynb`

**Objetivo:** Implementar una Red Neuronal Cuántica (QNN) y entrenarla para
clasificación binaria, discutiendo el problema del barren plateau.

**Estructura del notebook:**

1. **Arquitectura QNN.** El circuito combina:
   - **Capa de embedding** (ZZFeatureMap): codifica los datos x en el estado cuántico.
   - **Capa variacional** (RealAmplitudes): parámetros entrenables θ.
   - **Observable de salida**: valor esperado de Z⊗I como función de decisión.

2. **EstimatorQNN.** Se configura la QNN con Qiskit conectando el circuito,
   los parámetros de entrada (datos) y los parámetros de peso (entrenables).

3. **Entrenamiento con COBYLA.** Se usa `NeuralNetworkClassifier` con un
   callback que registra la pérdida en cada iteración.

4. **Curva de aprendizaje.** Se traza la pérdida en función de la iteración y
   se evalúa la exactitud en el conjunto de test.

5. **Barren plateau.** Se discute por qué la varianza del gradiente escala
   exponencialmente con el número de qubits y capas, y qué estrategias existen
   para mitigarlo (inicialización inteligente, ansatz estructurado).

6. **Ejercicios.** Cambiar a ADAM con parameter-shift. Aumentar capas del ansatz.
   Comparar con una red neuronal clásica equivalente.

**Módulos usados:** `src/visualization.py`

**Requiere:** `qiskit-machine-learning`

---

## Cómo añadir un nuevo capítulo

1. Crear el directorio `notebooks/ch09_<tema>/`.
2. Crear el notebook siguiendo la plantilla:
   - Celda Markdown: **Objetivos** (3-5 puntos concretos).
   - Celda de código: imports con `sys.path.insert`.
   - Alternar celdas Markdown (teoría con LaTeX) y código (implementación).
   - Celda final Markdown: **Ejercicios propuestos** (al menos 3).
3. Si el nuevo tema necesita utilidades matemáticas nuevas, añadirlas a `src/`
   con docstrings NumPy-style y sus tests correspondientes en `tests/`.
4. Actualizar `README.md` y `GUIA.md` con la nueva entrada.

---

## Hoja de ruta

| Prioridad | Extensión propuesta | Carpeta sugerida |
|---|---|---|
| Alta | Corrección de errores cuánticos (Steane, Shor 9-qubit) | `ch09_error_correction/` |
| Alta | Simulación de ruido (canales de Kraus, Lindblad) | `src/noise_models.py` |
| Media | Estimación de fase cuántica (QPE independiente) | `ch04_fourier_cuantica/02_qpe.ipynb` |
| Media | VQLS (Variational Quantum Linear Solver) | `ch07_variacional/04_vqls.ipynb` |
| Baja | Ejecución en hardware real IBM Quantum | `ch10_hardware/` |
| Baja | Backends alternativos (PennyLane, Cirq) | `src/backends/` |
