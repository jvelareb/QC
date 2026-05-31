"""
test_gates.py
=============
Tests de unitariedad y corrección algebraica para todos los módulos
de puertas cuánticas en src/quantum_gates.py.

Ejecutar con:
    pytest tests/test_gates.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.quantum_gates import Gates


# ---------------------------------------------------------------------------
# Unitariedad
# ---------------------------------------------------------------------------
FIXED_GATES = [Gates.I, Gates.X, Gates.Y, Gates.Z,
               Gates.H, Gates.S, Gates.T, Gates.Sdg, Gates.Tdg,
               Gates.CNOT, Gates.CZ, Gates.SWAP]

@pytest.mark.parametrize("gate", FIXED_GATES)
def test_unitarity_fixed_gates(gate):
    """Verifica que U†U = I para todas las puertas fijas."""
    assert Gates.is_unitary(gate), f"La puerta {gate} no es unitaria."


@pytest.mark.parametrize("theta", [0, np.pi/6, np.pi/4, np.pi/2, np.pi, 2*np.pi])
def test_unitarity_parametric(theta):
    """Verifica unitariedad de Rx, Ry, Rz para varios ángulos."""
    for gate_fn in [Gates.Rx, Gates.Ry, Gates.Rz]:
        gate = gate_fn(theta)
        assert Gates.is_unitary(gate), f"{gate_fn.__name__}({theta}) no es unitaria."


# ---------------------------------------------------------------------------
# Identidades algebraicas clave
# ---------------------------------------------------------------------------
def test_HH_is_identity():
    assert np.allclose(Gates.H @ Gates.H, Gates.I)

def test_XX_is_identity():
    assert np.allclose(Gates.X @ Gates.X, Gates.I)

def test_HXH_equals_Z():
    assert np.allclose(Gates.H @ Gates.X @ Gates.H, Gates.Z)

def test_HZH_equals_X():
    assert np.allclose(Gates.H @ Gates.Z @ Gates.H, Gates.X)

def test_SS_equals_Z():
    assert np.allclose(Gates.S @ Gates.S, Gates.Z)

def test_TT_equals_S():
    assert np.allclose(Gates.T @ Gates.T, Gates.S)

def test_T8_is_identity():
    T8 = np.linalg.matrix_power(Gates.T, 8)
    assert np.allclose(T8, Gates.I)

def test_S_Sdg_is_identity():
    assert np.allclose(Gates.S @ Gates.Sdg, Gates.I)


# ---------------------------------------------------------------------------
# Rotaciones
# ---------------------------------------------------------------------------
def test_Rx_pi_equals_iX():
    """Rx(π) = -iX (salvo fase global)."""
    Rx_pi = Gates.Rx(np.pi)
    ratio = Rx_pi / (-1j * Gates.X)
    assert np.allclose(np.abs(ratio), np.ones((2, 2))), \
        "Rx(π) no es proporcional a -iX"

def test_Rz_double_angle():
    """Rz(θ1 + θ2) = Rz(θ1) @ Rz(θ2)."""
    t1, t2 = np.pi/3, np.pi/5
    assert np.allclose(Gates.Rz(t1 + t2), Gates.Rz(t1) @ Gates.Rz(t2))


# ---------------------------------------------------------------------------
# Tensor product
# ---------------------------------------------------------------------------
def test_tensor_dimensions():
    HI = Gates.tensor(Gates.H, Gates.I)
    assert HI.shape == (4, 4)
    HIH = Gates.tensor(Gates.H, Gates.I, Gates.H)
    assert HIH.shape == (8, 8)

def test_tensor_unitarity():
    HI = Gates.tensor(Gates.H, Gates.I)
    assert Gates.is_unitary(HI)
