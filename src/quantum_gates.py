"""
quantum_gates.py
================
Módulo de puertas cuánticas representadas como matrices NumPy.

Contiene todas las puertas unitarias de uno y dos qubits utilizadas
a lo largo del libro, junto con métodos auxiliares para componer y
aplicar puertas sobre estados arbitrarios.

Autor: J. Velasco
Versión: 1.0.0
"""

import numpy as np
from typing import Union

# ---------------------------------------------------------------------------
# Tipo básico
# ---------------------------------------------------------------------------
Matrix = np.ndarray   # alias para legibilidad


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------
class Gates:
    """
    Colección estática de matrices de puertas cuánticas estándar.

    Todos los atributos son matrices NumPy de dtype complex128.
    Las puertas de un qubit son matrices 2×2; las de dos qubits, 4×4.

    Ejemplo de uso::

        from src.quantum_gates import Gates

        psi = np.array([1, 0], dtype=complex)   # |0>
        psi_h = Gates.H @ psi                   # Hadamard
    """

    # ------------------------------------------------------------------
    # Puertas de un qubit
    # ------------------------------------------------------------------
    I: Matrix = np.eye(2, dtype=complex)
    """Identidad 2×2."""

    X: Matrix = np.array([[0, 1],
                           [1, 0]], dtype=complex)
    """Puerta de Pauli X (NOT cuántico)."""

    Y: Matrix = np.array([[0, -1j],
                           [1j,  0]], dtype=complex)
    """Puerta de Pauli Y."""

    Z: Matrix = np.array([[1,  0],
                           [0, -1]], dtype=complex)
    """Puerta de Pauli Z (inversión de fase)."""

    H: Matrix = (1 / np.sqrt(2)) * np.array([[1,  1],
                                              [1, -1]], dtype=complex)
    """Puerta de Hadamard."""

    S: Matrix = np.array([[1, 0],
                           [0, 1j]], dtype=complex)
    """Puerta de fase S (sqrt(Z))."""

    T: Matrix = np.array([[1, 0],
                           [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    """Puerta T (pi/8)."""

    Sdg: Matrix = np.array([[1,  0],
                             [0, -1j]], dtype=complex)
    """Puerta S† (adjunta de S)."""

    Tdg: Matrix = np.array([[1, 0],
                             [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
    """Puerta T† (adjunta de T)."""

    # ------------------------------------------------------------------
    # Puertas de dos qubits
    # ------------------------------------------------------------------
    CNOT: Matrix = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0]], dtype=complex)
    """CNOT (control-NOT, también CX)."""

    CZ: Matrix = np.array([[1, 0, 0,  0],
                            [0, 1, 0,  0],
                            [0, 0, 1,  0],
                            [0, 0, 0, -1]], dtype=complex)
    """Puerta CZ (fase controlada)."""

    SWAP: Matrix = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]], dtype=complex)
    """Puerta SWAP."""

    # ------------------------------------------------------------------
    # Puertas parametrizadas (métodos estáticos)
    # ------------------------------------------------------------------
    @staticmethod
    def Rx(theta: float) -> Matrix:
        """Rotación en el eje X de la esfera de Bloch por ángulo theta.

        Parameters
        ----------
        theta : float
            Ángulo de rotación en radianes.

        Returns
        -------
        Matrix
            Matriz unitaria 2×2.
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c,     -1j * s],
                         [-1j * s, c    ]], dtype=complex)

    @staticmethod
    def Ry(theta: float) -> Matrix:
        """Rotación en el eje Y por ángulo theta."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s],
                         [s,  c]], dtype=complex)

    @staticmethod
    def Rz(theta: float) -> Matrix:
        """Rotación en el eje Z por ángulo theta."""
        return np.array([[np.exp(-1j * theta / 2), 0],
                         [0, np.exp(1j * theta / 2)]], dtype=complex)

    @staticmethod
    def P(phi: float) -> Matrix:
        """Puerta de fase P(phi): aplica e^{i*phi} al estado |1>."""
        return np.array([[1, 0],
                         [0, np.exp(1j * phi)]], dtype=complex)

    @staticmethod
    def CR(k: int) -> Matrix:
        """Puerta de rotación de fase controlada CR_k, usada en la QFT.

        Parameters
        ----------
        k : int
            Entero positivo; la fase aplicada es 2*pi / 2^k.
        """
        phi = 2 * np.pi / (2 ** k)
        return np.array([[1, 0, 0,              0],
                         [0, 1, 0,              0],
                         [0, 0, 1,              0],
                         [0, 0, 0, np.exp(1j * phi)]], dtype=complex)

    # ------------------------------------------------------------------
    # Métodos auxiliares de composición
    # ------------------------------------------------------------------
    @staticmethod
    def tensor(*matrices: Matrix) -> Matrix:
        """Producto tensorial de una secuencia de matrices (A ⊗ B ⊗ …).

        Parameters
        ----------
        *matrices : Matrix
            Matrices 2D a combinar con el producto de Kronecker.

        Returns
        -------
        Matrix
            Producto tensorial resultante.

        Ejemplo
        -------
        >>> H_I = Gates.tensor(Gates.H, Gates.I)  # H aplicada al qubit 0
        """
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result

    @staticmethod
    def is_unitary(U: Matrix, tol: float = 1e-10) -> bool:
        """Comprueba si la matriz U es unitaria (U†U = I).

        Parameters
        ----------
        U : Matrix
            Matriz cuadrada compleja.
        tol : float
            Tolerancia numérica para la comparación.

        Returns
        -------
        bool
        """
        n = U.shape[0]
        product = U.conj().T @ U
        return np.allclose(product, np.eye(n), atol=tol)

    @staticmethod
    def apply(gate: Matrix, state: np.ndarray) -> np.ndarray:
        """Aplica una puerta a un vector de estado.

        Parameters
        ----------
        gate : Matrix
            Matriz unitaria de la puerta.
        state : np.ndarray
            Vector de estado columna (normalizado).

        Returns
        -------
        np.ndarray
            Nuevo vector de estado.
        """
        return gate @ state
