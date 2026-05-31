"""
visualization.py
================
Unified visualization module for the book repository.

Provides high-level functions for:
  - Bar charts of probability distributions
  - Density matrix visualization
  - Bloch vector evolution (static)
  - Qiskit-style measurement result histograms
  - Unitary matrices (modulus and phase)

Requires matplotlib. Functions that use Plotly are optional.

Author: J. Velasco
Version: 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------
BOOK_STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.labelcolor": "#e6edf3",
    "axes.edgecolor": "#30363d",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#30363d",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
}


def set_book_style():
    """Applies the book's visual style to all matplotlib figures."""
    plt.rcParams.update(BOOK_STYLE)


# ---------------------------------------------------------------------------
# Count / probability histogram
# ---------------------------------------------------------------------------
class QuantumVisualization:
    """Collection of quantum visualization functions."""

    @staticmethod
    def plot_histogram(
        counts: Dict[str, int],
        title: str = "Measurement distribution",
        color: str = "#58a6ff",
        figsize: Tuple[int, int] = (9, 4),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Draws the histogram of a quantum measurement result.

        Parameters
        ----------
        counts : dict
            Dictionary {bitstring: count} as returned by Qiskit or
            QuantumMath.measure().
        title : str
            Chart title.
        color : str
            Bar color (hex or matplotlib name).
        figsize : tuple
            Figure size in inches.
        ax : Optional[plt.Axes]
            Existing axis to draw on (creates a new one if None).

        Returns
        -------
        plt.Figure
        """
        set_book_style()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        labels = list(counts.keys())
        values = list(counts.values())
        total = sum(values)
        probs = [v / total for v in values]

        bars = ax.bar(labels, probs, color=color, alpha=0.85,
                      edgecolor="#1f6feb", linewidth=0.8)

        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{prob:.3f}",
                    ha="center", va="bottom", fontsize=9, color="#e6edf3")

        ax.set_xlabel("Measured state")
        ax.set_ylabel("Probability")
        ax.set_title(title)
        ax.set_ylim(0, max(probs) * 1.2)
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    @staticmethod
    def plot_bloch_vector(
        state: np.ndarray,
        title: str = "Bloch Vector",
        figsize: Tuple[int, int] = (5, 5),
    ) -> plt.Figure:
        """Draws the Bloch vector of a pure single-qubit state.

        Parameters
        ----------
        state : np.ndarray
            Complex state vector [alpha, beta] of 1 qubit.
        title : str
            Chart title.
        figsize : tuple
            Figure size.

        Returns
        -------
        plt.Figure
        """
        from .quantum_math import QuantumMath

        x, y, z = QuantumMath.bloch_vector(state)

        set_book_style()
        fig = plt.figure(figsize=figsize, facecolor="#0d1117")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#161b22")

        # Bloch sphere (wireframe)
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 40)
        sx = np.outer(np.cos(u), np.sin(v))
        sy = np.outer(np.sin(u), np.sin(v))
        sz = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(sx, sy, sz, color="#30363d", alpha=0.25, linewidth=0.5)

        # Axes
        ax.quiver(0, 0, 0, 1.3, 0, 0, color="#8b949e", linewidth=0.8, arrow_length_ratio=0.07)
        ax.quiver(0, 0, 0, 0, 1.3, 0, color="#8b949e", linewidth=0.8, arrow_length_ratio=0.07)
        ax.quiver(0, 0, 0, 0, 0, 1.3, color="#8b949e", linewidth=0.8, arrow_length_ratio=0.07)
        for label, pos in [("X", (1.4, 0, 0)), ("Y", (0, 1.4, 0)),
                            ("Z", (0, 0, 1.4)), ("|0⟩", (0, 0, 1.05)),
                            ("|1⟩", (0, 0, -1.15))]:
            ax.text(*pos, label, color="#e6edf3", fontsize=10, ha="center")

        # State vector
        ax.quiver(0, 0, 0, x, y, z, color="#f78166", linewidth=2,
                  arrow_length_ratio=0.12)
        ax.scatter([x], [y], [z], color="#f78166", s=40, zorder=5)

        # Dotted projection
        ax.plot([x, x], [y, y], [0, z], linestyle="--",
                color="#58a6ff", alpha=0.5, linewidth=0.9)

        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_zlim([-1.3, 1.3])
        ax.set_title(title, color="#e6edf3", pad=10)
        ax.set_axis_off()
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    @staticmethod
    def plot_unitary(
        U: np.ndarray,
        title: str = "Unitary Matrix",
        figsize: Tuple[int, int] = (10, 4),
    ) -> plt.Figure:
        """Visualizes the modulus and phase of a unitary matrix.

        Parameters
        ----------
        U : np.ndarray
            Complex unitary matrix.
        title : str
            General figure title.
        figsize : tuple
            Figure size.

        Returns
        -------
        plt.Figure
        """
        set_book_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        n = U.shape[0]
        ticks = [format(i, f"0{int(np.log2(n))}b") for i in range(n)]

        # Modulus
        im1 = ax1.imshow(np.abs(U), cmap="Blues",
                         vmin=0, vmax=1, aspect="auto")
        ax1.set_title(f"{title} — Modulus |U|")
        ax1.set_xticks(range(n))
        ax1.set_xticklabels(ticks, fontsize=7, rotation=45)
        ax1.set_yticks(range(n))
        ax1.set_yticklabels(ticks, fontsize=7)
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # Phase
        im2 = ax2.imshow(np.angle(U), cmap="hsv",
                         vmin=-np.pi, vmax=np.pi, aspect="auto")
        ax2.set_title(f"{title} — Phase arg(U)")
        ax2.set_xticks(range(n))
        ax2.set_xticklabels(ticks, fontsize=7, rotation=45)
        ax2.set_yticks(range(n))
        ax2.set_yticklabels(ticks, fontsize=7)
        plt.colorbar(im2, ax=ax2, fraction=0.046, label="radians")

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    @staticmethod
    def plot_state_vector(
        state: np.ndarray,
        title: str = "State vector",
        figsize: Tuple[int, int] = (9, 4),
    ) -> plt.Figure:
        """Visualizes the amplitudes (modulus and phase) of a state vector.

        Parameters
        ----------
        state : np.ndarray
            Normalized state vector (length = 2^n).
        title : str
            Chart title.
        figsize : tuple
            Figure size.

        Returns
        -------
        plt.Figure
        """
        set_book_style()
        n_qubits = int(np.log2(len(state)))
        labels = [format(i, f"0{n_qubits}b") for i in range(len(state))]
        probs = np.abs(state) ** 2
        phases = np.angle(state)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Probabilities
        colors = plt.cm.plasma(probs / (probs.max() + 1e-14))
        ax1.bar(labels, probs, color=colors, edgecolor="#1f6feb", alpha=0.9)
        ax1.set_xlabel("Basis state")
        ax1.set_ylabel("Probability")
        ax1.set_title(f"{title} — Probabilities")
        ax1.tick_params(axis="x", rotation=60)
        ax1.grid(axis="y", alpha=0.4)

        # Phases
        mask = probs > 1e-10
        ax2.bar(np.array(labels)[mask],
                np.degrees(phases[mask]),
                color="#f78166", edgecolor="#da3633", alpha=0.9)
        ax2.set_xlabel("Basis state")
        ax2.set_ylabel("Phase (degrees)")
        ax2.set_title(f"{title} — Phases")
        ax2.tick_params(axis="x", rotation=60)
        ax2.axhline(0, color="#8b949e", linewidth=0.7)
        ax2.grid(axis="y", alpha=0.4)

        plt.tight_layout()
        return fig
