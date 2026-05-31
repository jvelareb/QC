"""
Módulo src — Computación Cuántica: Repositorio del Libro
Exporta las principales clases y funciones de los submódulos.
"""
from .quantum_gates import Gates
from .quantum_math import QuantumMath
from .visualization import QuantumVisualization

__version__ = "1.0.0"
__all__ = ["Gates", "QuantumMath", "QuantumVisualization"]
