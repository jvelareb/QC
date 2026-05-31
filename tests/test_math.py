"""
test_math.py
============
Tests de corrección para las utilidades matemáticas cuánticas
en src/quantum_math.py.

Ejecutar con:
    pytest tests/test_math.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.quantum_math import QuantumMath


# ---------------------------------------------------------------------------
# Estados básicos
# ---------------------------------------------------------------------------
def test_ket0_normalized():
    assert np.isclose(np.linalg.norm(QuantumMath.ket0()), 1.0)

def test_ket1_normalized():
    assert np.isclose(np.linalg.norm(QuantumMath.ket1()), 1.0)

def test_ket_plus_normalized():
    assert np.isclose(np.linalg.norm(QuantumMath.ket_plus()), 1.0)

def test_ket_minus_normalized():
    assert np.isclose(np.linalg.norm(QuantumMath.ket_minus()), 1.0)

def test_ket0_ket1_orthogonal():
    inner = QuantumMath.inner_product(QuantumMath.ket0(), QuantumMath.ket1())
    assert np.isclose(inner, 0.0)

def test_ket_plus_minus_orthogonal():
    inner = QuantumMath.inner_product(QuantumMath.ket_plus(), QuantumMath.ket_minus())
    assert np.isclose(inner, 0.0)


# ---------------------------------------------------------------------------
# Probabilidades
# ---------------------------------------------------------------------------
def test_probabilities_sum_to_one():
    from src.quantum_gates import Gates
    state = Gates.H @ QuantumMath.ket0()
    probs = QuantumMath.probabilities(state)
    assert np.isclose(np.sum(probs), 1.0)

def test_ket0_prob_distribution():
    probs = QuantumMath.probabilities(QuantumMath.ket0())
    assert np.isclose(probs[0], 1.0) and np.isclose(probs[1], 0.0)


# ---------------------------------------------------------------------------
# Tensor product
# ---------------------------------------------------------------------------
def test_tensor_dimension():
    state = QuantumMath.tensor_product(QuantumMath.ket0(), QuantumMath.ket1())
    assert len(state) == 4

def test_tensor_normalization():
    state = QuantumMath.tensor_product(QuantumMath.ket_plus(), QuantumMath.ket0())
    assert np.isclose(np.linalg.norm(state), 1.0)


# ---------------------------------------------------------------------------
# Bloch vector
# ---------------------------------------------------------------------------
def test_bloch_ket0_is_north_pole():
    x, y, z = QuantumMath.bloch_vector(QuantumMath.ket0())
    assert np.isclose(x, 0.0) and np.isclose(y, 0.0) and np.isclose(z, 1.0)

def test_bloch_ket1_is_south_pole():
    x, y, z = QuantumMath.bloch_vector(QuantumMath.ket1())
    assert np.isclose(x, 0.0) and np.isclose(y, 0.0) and np.isclose(z, -1.0)

def test_bloch_norm_pure_state():
    from src.quantum_gates import Gates
    state = Gates.H @ QuantumMath.ket0()
    bv = QuantumMath.bloch_vector(state)
    norm = np.sqrt(sum(c**2 for c in bv))
    assert np.isclose(norm, 1.0)


# ---------------------------------------------------------------------------
# Fidelidad
# ---------------------------------------------------------------------------
def test_fidelity_same_state():
    state = QuantumMath.ket_plus()
    assert np.isclose(QuantumMath.fidelity(state, state), 1.0)

def test_fidelity_orthogonal_states():
    assert np.isclose(QuantumMath.fidelity(QuantumMath.ket0(), QuantumMath.ket1()), 0.0)

def test_fidelity_bounds():
    from src.quantum_gates import Gates
    s1 = QuantumMath.ket0()
    s2 = Gates.H @ s1
    f = QuantumMath.fidelity(s1, s2)
    assert 0.0 <= f <= 1.0


# ---------------------------------------------------------------------------
# Von Neumann entropy
# ---------------------------------------------------------------------------
def test_entropy_pure_state_is_zero():
    rho = QuantumMath.density_matrix(QuantumMath.ket0())
    assert np.isclose(QuantumMath.von_neumann_entropy(rho), 0.0, atol=1e-10)

def test_entropy_maximally_mixed():
    rho_mixed = np.eye(2, dtype=complex) / 2
    entropy = QuantumMath.von_neumann_entropy(rho_mixed)
    assert np.isclose(entropy, 1.0)


# ---------------------------------------------------------------------------
# QFT matrix
# ---------------------------------------------------------------------------
def test_qft_matrix_unitarity():
    for n in [1, 2, 3, 4]:
        U = QuantumMath.qft_matrix(n)
        UdU = U.conj().T @ U
        assert np.allclose(UdU, np.eye(2**n), atol=1e-10), \
            f"QFT({n}) no es unitaria"

def test_qft_matrix_dimension():
    for n in [2, 3, 4]:
        U = QuantumMath.qft_matrix(n)
        assert U.shape == (2**n, 2**n)
