"""
quantum_gates.py
================
Module of quantum gates represented as NumPy matrices.

Contains all one- and two-qubit unitary gates used
throughout the book, along with helper methods to compose and
apply gates to arbitrary states.

Author: J. Velasco
Version: 1.0.0
"""

import numpy as np
from typing import Union

# ---------------------------------------------------------------------------
# Basic type
# ---------------------------------------------------------------------------
Matrix = np.ndarray   # alias for readability


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class Gates:
    """
    Static collection of standard quantum gate matrices.

    All attributes are NumPy matrices of dtype complex128.
    Single-qubit gates are 2x2 matrices; two-qubit gates are 4x4.

    Usage example::

        from src.quantum_gates import Gates

        psi = np.array([1, 0], dtype=complex)   # |0>
        psi_h = Gates.H @ psi                   # Hadamard
    """

    # ------------------------------------------------------------------
    # Single-qubit gates
    # ------------------------------------------------------------------
    I: Matrix = np.eye(2, dtype=complex)
    """2x2 Identity."""

    X: Matrix = np.array([[0, 1],
                           [1, 0]], dtype=complex)
    """Pauli X gate (quantum NOT)."""

    Y: Matrix = np.array([[0, -1j],
                           [1j,  0]], dtype=complex)
    """Pauli Y gate."""

    Z: Matrix = np.array([[1,  0],
                           [0, -1]], dtype=complex)
    """Pauli Z gate (phase flip)."""

    H: Matrix = (1 / np.sqrt(2)) * np.array([[1,  1],
                                              [1, -1]], dtype=complex)
    """Hadamard gate."""

    S: Matrix = np.array([[1, 0],
                           [0, 1j]], dtype=complex)
    """S phase gate (sqrt(Z))."""

    T: Matrix = np.array([[1, 0],
                           [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    """T gate (pi/8)."""

    Sdg: Matrix = np.array([[1,  0],
                             [0, -1j]], dtype=complex)
    """S† gate (adjoint of S)."""

    Tdg: Matrix = np.array([[1, 0],
                             [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
    """T† gate (adjoint of T)."""

    # ------------------------------------------------------------------
    # Two-qubit gates
    # ------------------------------------------------------------------
    CNOT: Matrix = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0]], dtype=complex)
    """CNOT (control-NOT, also CX)."""

    CZ: Matrix = np.array([[1, 0, 0,  0],
                            [0, 1, 0,  0],
                            [0, 0, 1,  0],
                            [0, 0, 0, -1]], dtype=complex)
    """CZ gate (controlled phase)."""

    SWAP: Matrix = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]], dtype=complex)
    """SWAP gate."""

    # ------------------------------------------------------------------
    # Parameterized gates (static methods)
    # ------------------------------------------------------------------
    @staticmethod
    def Rx(theta: float) -> Matrix:
        """Rotation around the X axis of the Bloch sphere by angle theta.

        Parameters
        ----------
        theta : float
            Rotation angle in radians.

        Returns
        -------
        Matrix
            2x2 unitary matrix.
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c,     -1j * s],
                         [-1j * s, c    ]], dtype=complex)

    @staticmethod
    def Ry(theta: float) -> Matrix:
        """Rotation around the Y axis by angle theta."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s],
                         [s,  c]], dtype=complex)

    @staticmethod
    def Rz(theta: float) -> Matrix:
        """Rotation around the Z axis by angle theta."""
        return np.array([[np.exp(-1j * theta / 2), 0],
                         [0, np.exp(1j * theta / 2)]], dtype=complex)

    @staticmethod
    def P(phi: float) -> Matrix:
        """Phase gate P(phi): applies e^{i*phi} to the |1> state."""
        return np.array([[1, 0],
                         [0, np.exp(1j * phi)]], dtype=complex)

    @staticmethod
    def CR(k: int) -> Matrix:
        """Controlled phase rotation gate CR_k, used in the QFT.

        Parameters
        ----------
        k : int
            Positive integer; the applied phase is 2*pi / 2^k.
        """
        phi = 2 * np.pi / (2 ** k)
        return np.array([[1, 0, 0,              0],
                         [0, 1, 0,              0],
                         [0, 0, 1,              0],
                         [0, 0, 0, np.exp(1j * phi)]], dtype=complex)

    # ------------------------------------------------------------------
    # Composition helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def tensor(*matrices: Matrix) -> Matrix:
        """Tensor product of a sequence of matrices (A x B x ...).

        Parameters
        ----------
        *matrices : Matrix
            2D matrices to combine with the Kronecker product.

        Returns
        -------
        Matrix
            Resulting tensor product.

        Example
        -------
        >>> H_I = Gates.tensor(Gates.H, Gates.I)  # H applied to qubit 0
        """
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result

    @staticmethod
    def is_unitary(U: Matrix, tol: float = 1e-10) -> bool:
        """Checks whether matrix U is unitary (U†U = I).

        Parameters
        ----------
        U : Matrix
            Square complex matrix.
        tol : float
            Numerical tolerance for comparison.

        Returns
        -------
        bool
        """
        n = U.shape[0]
        product = U.conj().T @ U
        return np.allclose(product, np.eye(n), atol=tol)

    @staticmethod
    def apply(gate: Matrix, state: np.ndarray) -> np.ndarray:
        """Applies a gate to a state vector.

        Parameters
        ----------
        gate : Matrix
            Unitary matrix of the gate.
        state : np.ndarray
            Column state vector (normalized).

        Returns
        -------
        np.ndarray
            New state vector.
        """
        return gate @ state
