"""
quantum_math.py
===============
Funciones de álgebra lineal y formalismo cuántico utilizadas como
utilidades matemáticas transversales a todo el libro.

Incluye:
  - Construcción y verificación de estados cuánticos
  - Cálculo de valores esperados y probabilidades
  - Descomposición espectral y fidelidad
  - Representación matricial de operadores
  - QFT analítica (referencia para comparación)

Autor: J. Velasco
Versión: 1.0.0
"""

import numpy as np
from typing import Union, List, Tuple
from numpy.linalg import eigh, svd, norm


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------
class QuantumMath:
    """Colección de herramientas matemáticas cuánticas."""

    # ------------------------------------------------------------------
    # Estados básicos
    # ------------------------------------------------------------------
    @staticmethod
    def ket0() -> np.ndarray:
        """Devuelve el estado base |0>."""
        return np.array([1, 0], dtype=complex)

    @staticmethod
    def ket1() -> np.ndarray:
        """Devuelve el estado base |1>."""
        return np.array([0, 1], dtype=complex)

    @staticmethod
    def ket_plus() -> np.ndarray:
        """Devuelve (|0> + |1>) / sqrt(2)."""
        return np.array([1, 1], dtype=complex) / np.sqrt(2)

    @staticmethod
    def ket_minus() -> np.ndarray:
        """Devuelve (|0> - |1>) / sqrt(2)."""
        return np.array([1, -1], dtype=complex) / np.sqrt(2)

    @staticmethod
    def computational_basis(n: int) -> List[np.ndarray]:
        """Genera la base computacional para un sistema de n qubits.

        Parameters
        ----------
        n : int
            Número de qubits.

        Returns
        -------
        List[np.ndarray]
            Lista de 2^n vectores de estado de la base computacional,
            en orden lexicográfico |00…0>, |00…1>, …, |11…1>.
        """
        dim = 2 ** n
        return [np.eye(dim, dtype=complex)[k] for k in range(dim)]

    # ------------------------------------------------------------------
    # Operaciones sobre estados
    # ------------------------------------------------------------------
    @staticmethod
    def tensor_product(*states: np.ndarray) -> np.ndarray:
        """Producto tensorial de varios estados (|ψ_1> ⊗ |ψ_2> ⊗ …).

        Parameters
        ----------
        *states : np.ndarray
            Vectores de estado individuales.

        Returns
        -------
        np.ndarray
            Estado compuesto.
        """
        result = states[0]
        for s in states[1:]:
            result = np.kron(result, s)
        return result

    @staticmethod
    def normalize(state: np.ndarray) -> np.ndarray:
        """Normaliza un vector de estado.

        Parameters
        ----------
        state : np.ndarray
            Vector de estado (puede no estar normalizado).

        Returns
        -------
        np.ndarray
            Vector normalizado.

        Raises
        ------
        ValueError
            Si el vector tiene norma cero.
        """
        n = norm(state)
        if n < 1e-14:
            raise ValueError("El estado tiene norma cero y no puede normalizarse.")
        return state / n

    @staticmethod
    def inner_product(bra: np.ndarray, ket: np.ndarray) -> complex:
        """Calcula el producto interior <bra|ket>.

        Parameters
        ----------
        bra : np.ndarray
            Vector de estado (se conjuga internamente).
        ket : np.ndarray
            Vector de estado.

        Returns
        -------
        complex
            Número complejo <bra|ket>.
        """
        return np.dot(bra.conj(), ket)

    @staticmethod
    def outer_product(ket: np.ndarray, bra: np.ndarray) -> np.ndarray:
        """Calcula el operador |ket><bra|.

        Parameters
        ----------
        ket : np.ndarray
            Vector columna.
        bra : np.ndarray
            Vector columna (se conjuga y transpone).

        Returns
        -------
        np.ndarray
            Matriz densidad o proyector.
        """
        return np.outer(ket, bra.conj())

    @staticmethod
    def density_matrix(state: np.ndarray) -> np.ndarray:
        """Calcula la matriz densidad rho = |ψ><ψ|.

        Parameters
        ----------
        state : np.ndarray
            Estado puro normalizado.

        Returns
        -------
        np.ndarray
            Matriz densidad (hermítica, semidefinida positiva, traza 1).
        """
        return np.outer(state, state.conj())

    # ------------------------------------------------------------------
    # Medida y probabilidades
    # ------------------------------------------------------------------
    @staticmethod
    def probabilities(state: np.ndarray) -> np.ndarray:
        """Calcula el vector de probabilidades de medida en la base computacional.

        Parameters
        ----------
        state : np.ndarray
            Vector de estado normalizado.

        Returns
        -------
        np.ndarray
            Vector de probabilidades (suma = 1).
        """
        return np.abs(state) ** 2

    @staticmethod
    def expectation_value(state: np.ndarray, operator: np.ndarray) -> float:
        """Calcula el valor esperado <ψ|O|ψ>.

        Parameters
        ----------
        state : np.ndarray
            Vector de estado normalizado.
        operator : np.ndarray
            Operador hermítico.

        Returns
        -------
        float
            Valor esperado (parte real, ya que el operador es hermítico).
        """
        return np.real(state.conj() @ operator @ state)

    @staticmethod
    def measure(state: np.ndarray, n_shots: int = 1024) -> dict:
        """Simula medidas sobre un estado cuántico.

        Parameters
        ----------
        state : np.ndarray
            Vector de estado normalizado de n qubits.
        n_shots : int
            Número de mediciones a realizar.

        Returns
        -------
        dict
            Diccionario {bitstring: conteo}.
        """
        probs = np.abs(state) ** 2
        n_qubits = int(np.log2(len(state)))
        indices = np.random.choice(len(state), size=n_shots, p=probs)
        counts = {}
        for idx in indices:
            bitstring = format(idx, f"0{n_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------------
    # Métricas de estado
    # ------------------------------------------------------------------
    @staticmethod
    def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        """Calcula la fidelidad F = |<ψ1|ψ2>|^2 entre dos estados puros.

        Parameters
        ----------
        state1, state2 : np.ndarray
            Vectores de estado normalizados.

        Returns
        -------
        float
            Fidelidad en [0, 1]. F=1 implica estados idénticos.
        """
        return float(np.abs(np.dot(state1.conj(), state2)) ** 2)

    @staticmethod
    def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
        """Calcula la distancia de traza T(rho, sigma) = (1/2) Tr|rho - sigma|.

        Parameters
        ----------
        rho, sigma : np.ndarray
            Matrices densidad.

        Returns
        -------
        float
            Distancia de traza en [0, 1].
        """
        delta = rho - sigma
        singular_values = svd(delta, compute_uv=False)
        return 0.5 * float(np.sum(singular_values))

    @staticmethod
    def von_neumann_entropy(rho: np.ndarray) -> float:
        """Calcula la entropía de Von Neumann S(rho) = -Tr(rho log rho).

        Parameters
        ----------
        rho : np.ndarray
            Matriz densidad.

        Returns
        -------
        float
            Entropía en bits (log base 2). 0 para estados puros.
        """
        eigenvalues = eigh(rho)[0]
        eigenvalues = eigenvalues[eigenvalues > 1e-14]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    # ------------------------------------------------------------------
    # Coordenadas de Bloch
    # ------------------------------------------------------------------
    @staticmethod
    def bloch_vector(state: np.ndarray) -> Tuple[float, float, float]:
        """Calcula el vector de Bloch (x, y, z) para un estado puro de 1 qubit.

        Parameters
        ----------
        state : np.ndarray
            Vector de estado de un qubit [alpha, beta].

        Returns
        -------
        Tuple[float, float, float]
            Coordenadas (x, y, z) en la esfera de Bloch. La norma es 1
            para estados puros.
        """
        alpha, beta = state[0], state[1]
        x = 2 * np.real(alpha.conj() * beta)
        y = 2 * np.imag(alpha.conj() * beta)
        z = float(np.abs(alpha) ** 2 - np.abs(beta) ** 2)
        return (float(x), float(y), float(z))

    @staticmethod
    def bloch_angles(state: np.ndarray) -> Tuple[float, float]:
        """Extrae los ángulos esféricos (theta, phi) del vector de Bloch.

        Parameters
        ----------
        state : np.ndarray
            Vector de estado de un qubit.

        Returns
        -------
        Tuple[float, float]
            theta en [0, pi], phi en [0, 2*pi].
        """
        alpha, beta = state[0], state[1]
        theta = 2 * np.arccos(np.clip(np.abs(alpha), 0, 1))
        phi = np.angle(beta) - np.angle(alpha)
        if phi < 0:
            phi += 2 * np.pi
        return (float(theta), float(phi))

    # ------------------------------------------------------------------
    # QFT analítica
    # ------------------------------------------------------------------
    @staticmethod
    def qft_matrix(n: int) -> np.ndarray:
        """Construye la matriz unitaria de la QFT para n qubits.

        La QFT de n qubits es la transformada de Fourier discreta sobre
        Z_{2^n}, definida como

            QFT|j> = (1/sqrt(N)) sum_{k=0}^{N-1} e^{2*pi*i*j*k/N} |k>

        con N = 2^n.

        Parameters
        ----------
        n : int
            Número de qubits.

        Returns
        -------
        np.ndarray
            Matriz unitaria (2^n × 2^n) de tipo complex128.
        """
        N = 2 ** n
        omega = np.exp(2j * np.pi / N)
        j_idx = np.arange(N)
        k_idx = np.arange(N)
        return (omega ** np.outer(j_idx, k_idx)) / np.sqrt(N)
