"""
quantum_math.py
===============
Linear algebra functions and quantum formalism used as
mathematical utilities shared across the entire book.

Includes:
  - Construction and verification of quantum states
  - Calculation of expected values and probabilities
  - Spectral decomposition and fidelity
  - Matrix representation of operators
  - Analytical QFT (reference for comparison)

Author: J. Velasco
Version: 1.0.0
"""

import numpy as np
from typing import Union, List, Tuple
from numpy.linalg import eigh, svd, norm


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class QuantumMath:
    """Collection of quantum mathematical tools."""

    # ------------------------------------------------------------------
    # Basic states
    # ------------------------------------------------------------------
    @staticmethod
    def ket0() -> np.ndarray:
        """Returns the basis state |0>."""
        return np.array([1, 0], dtype=complex)

    @staticmethod
    def ket1() -> np.ndarray:
        """Returns the basis state |1>."""
        return np.array([0, 1], dtype=complex)

    @staticmethod
    def ket_plus() -> np.ndarray:
        """Returns (|0> + |1>) / sqrt(2)."""
        return np.array([1, 1], dtype=complex) / np.sqrt(2)

    @staticmethod
    def ket_minus() -> np.ndarray:
        """Returns (|0> - |1>) / sqrt(2)."""
        return np.array([1, -1], dtype=complex) / np.sqrt(2)

    @staticmethod
    def computational_basis(n: int) -> List[np.ndarray]:
        """Generates the computational basis for an n-qubit system.

        Parameters
        ----------
        n : int
            Number of qubits.

        Returns
        -------
        List[np.ndarray]
            List of 2^n basis state vectors from the computational basis,
            in lexicographic order |00...0>, |00...1>, ..., |11...1>.
        """
        dim = 2 ** n
        return [np.eye(dim, dtype=complex)[k] for k in range(dim)]

    # ------------------------------------------------------------------
    # State operations
    # ------------------------------------------------------------------
    @staticmethod
    def tensor_product(*states: np.ndarray) -> np.ndarray:
        """Tensor product of multiple states (|psi_1> tensor |psi_2> tensor ...).

        Parameters
        ----------
        *states : np.ndarray
            Individual state vectors.

        Returns
        -------
        np.ndarray
            Composite state.
        """
        result = states[0]
        for s in states[1:]:
            result = np.kron(result, s)
        return result

    @staticmethod
    def normalize(state: np.ndarray) -> np.ndarray:
        """Normalizes a state vector.

        Parameters
        ----------
        state : np.ndarray
            State vector (may not be normalized).

        Returns
        -------
        np.ndarray
            Normalized vector.

        Raises
        ------
        ValueError
            If the vector has zero norm.
        """
        n = norm(state)
        if n < 1e-14:
            raise ValueError("The state has zero norm and cannot be normalized.")
        return state / n

    @staticmethod
    def inner_product(bra: np.ndarray, ket: np.ndarray) -> complex:
        """Calculates the inner product <bra|ket>.

        Parameters
        ----------
        bra : np.ndarray
            State vector (conjugated internally).
        ket : np.ndarray
            State vector.

        Returns
        -------
        complex
            Complex number <bra|ket>.
        """
        return np.dot(bra.conj(), ket)

    @staticmethod
    def outer_product(ket: np.ndarray, bra: np.ndarray) -> np.ndarray:
        """Calculates the operator |ket><bra|.

        Parameters
        ----------
        ket : np.ndarray
            Column vector.
        bra : np.ndarray
            Column vector (conjugated and transposed).

        Returns
        -------
        np.ndarray
            Density matrix or projector.
        """
        return np.outer(ket, bra.conj())

    @staticmethod
    def density_matrix(state: np.ndarray) -> np.ndarray:
        """Calculates the density matrix rho = |psi><psi|.

        Parameters
        ----------
        state : np.ndarray
            Normalized pure state.

        Returns
        -------
        np.ndarray
            Density matrix (Hermitian, positive semidefinite, trace 1).
        """
        return np.outer(state, state.conj())

    # ------------------------------------------------------------------
    # Measurement and probabilities
    # ------------------------------------------------------------------
    @staticmethod
    def probabilities(state: np.ndarray) -> np.ndarray:
        """Calculates the measurement probability vector in the computational basis.

        Parameters
        ----------
        state : np.ndarray
            Normalized state vector.

        Returns
        -------
        np.ndarray
            Probability vector (sum = 1).
        """
        return np.abs(state) ** 2

    @staticmethod
    def expectation_value(state: np.ndarray, operator: np.ndarray) -> float:
        """Calculates the expected value <psi|O|psi>.

        Parameters
        ----------
        state : np.ndarray
            Normalized state vector.
        operator : np.ndarray
            Hermitian operator.

        Returns
        -------
        float
            Expected value (real part, since the operator is Hermitian).
        """
        return np.real(state.conj() @ operator @ state)

    @staticmethod
    def measure(state: np.ndarray, n_shots: int = 1024) -> dict:
        """Simulates measurements on a quantum state.

        Parameters
        ----------
        state : np.ndarray
            Normalized state vector of n qubits.
        n_shots : int
            Number of measurements to perform.

        Returns
        -------
        dict
            Dictionary {bitstring: count}.
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
    # State metrics
    # ------------------------------------------------------------------
    @staticmethod
    def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        """Calculates the fidelity F = |<psi1|psi2>|^2 between two pure states.

        Parameters
        ----------
        state1, state2 : np.ndarray
            Normalized state vectors.

        Returns
        -------
        float
            Fidelity in [0, 1]. F=1 implies identical states.
        """
        return float(np.abs(np.dot(state1.conj(), state2)) ** 2)

    @staticmethod
    def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
        """Calculates the trace distance T(rho, sigma) = (1/2) Tr|rho - sigma|.

        Parameters
        ----------
        rho, sigma : np.ndarray
            Density matrices.

        Returns
        -------
        float
            Trace distance in [0, 1].
        """
        delta = rho - sigma
        singular_values = svd(delta, compute_uv=False)
        return 0.5 * float(np.sum(singular_values))

    @staticmethod
    def von_neumann_entropy(rho: np.ndarray) -> float:
        """Calculates the Von Neumann entropy S(rho) = -Tr(rho log rho).

        Parameters
        ----------
        rho : np.ndarray
            Density matrix.

        Returns
        -------
        float
            Entropy in bits (log base 2). 0 for pure states.
        """
        eigenvalues = eigh(rho)[0]
        eigenvalues = eigenvalues[eigenvalues > 1e-14]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    # ------------------------------------------------------------------
    # Bloch coordinates
    # ------------------------------------------------------------------
    @staticmethod
    def bloch_vector(state: np.ndarray) -> Tuple[float, float, float]:
        """Calculates the Bloch vector (x, y, z) for a pure 1-qubit state.

        Parameters
        ----------
        state : np.ndarray
            State vector of one qubit [alpha, beta].

        Returns
        -------
        Tuple[float, float, float]
            Coordinates (x, y, z) on the Bloch sphere. The norm is 1
            for pure states.
        """
        alpha, beta = state[0], state[1]
        x = 2 * np.real(alpha.conj() * beta)
        y = 2 * np.imag(alpha.conj() * beta)
        z = float(np.abs(alpha) ** 2 - np.abs(beta) ** 2)
        return (float(x), float(y), float(z))

    @staticmethod
    def bloch_angles(state: np.ndarray) -> Tuple[float, float]:
        """Extracts the spherical angles (theta, phi) from the Bloch vector.

        Parameters
        ----------
        state : np.ndarray
            State vector of one qubit.

        Returns
        -------
        Tuple[float, float]
            theta in [0, pi], phi in [0, 2*pi].
        """
        alpha, beta = state[0], state[1]
        theta = 2 * np.arccos(np.clip(np.abs(alpha), 0, 1))
        phi = np.angle(beta) - np.angle(alpha)
        if phi < 0:
            phi += 2 * np.pi
        return (float(theta), float(phi))

    # ------------------------------------------------------------------
    # Analytical QFT
    # ------------------------------------------------------------------
    @staticmethod
    def qft_matrix(n: int) -> np.ndarray:
        """Constructs the unitary QFT matrix for n qubits.

        The QFT for n qubits is the discrete Fourier transform over
        Z_{2^n}, defined as

            QFT|j> = (1/sqrt(N)) sum_{k=0}^{N-1} e^{2*pi*i*j*k/N} |k>

        with N = 2^n.

        Parameters
        ----------
        n : int
            Number of qubits.

        Returns
        -------
        np.ndarray
            Unitary matrix (2^n x 2^n) of type complex128.
        """
        N = 2 ** n
        omega = np.exp(2j * np.pi / N)
        j_idx = np.arange(N)
        k_idx = np.arange(N)
        return (omega ** np.outer(j_idx, k_idx)) / np.sqrt(N)
