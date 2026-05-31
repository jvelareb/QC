"""
bloch_simulator.py
==================
Bloch sphere simulator — NumPy + Plotly.

Allows applying sequences of quantum gates to an initial qubit
and visualizing the resulting Bloch vector trajectory.

Designed for use from Jupyter notebooks with ipywidgets.
See: notebooks/ch01_introduccion/02_bloch_interactivo.ipynb

Author: J. Velasco
Version: 1.1.0
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Optional
from .quantum_gates import Gates
from .quantum_math import QuantumMath


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class BlochSimulator:
    """
    Interactive single-qubit simulator on the Bloch sphere.

    Maintains the internal qubit state and records the trajectory
    of the Bloch vector after each operation, allowing the
    evolution to be animated.

    Usage example::

        sim = BlochSimulator()
        sim.apply_gate("H")
        sim.apply_gate("Rx", theta=np.pi/4)
        fig = sim.plot_trajectory()
        fig.show()
    """

    GATES = {
        "I":   Gates.I,
        "X":   Gates.X,
        "Y":   Gates.Y,
        "Z":   Gates.Z,
        "H":   Gates.H,
        "S":   Gates.S,
        "T":   Gates.T,
        "Sdg": Gates.Sdg,
        "Tdg": Gates.Tdg,
    }

    PARAM_GATES = {"Rx", "Ry", "Rz", "P"}

    INITIAL_STATES = {
        "|0⟩":  np.array([1, 0], dtype=complex),
        "|1⟩":  np.array([0, 1], dtype=complex),
        "|+⟩":  np.array([1, 1], dtype=complex) / np.sqrt(2),
        "|-⟩":  np.array([1, -1], dtype=complex) / np.sqrt(2),
        "|i⟩":  np.array([1, 1j], dtype=complex) / np.sqrt(2),
        "|-i⟩": np.array([1, -1j], dtype=complex) / np.sqrt(2),
    }

    def __init__(self, initial_state: str = "|0⟩"):
        """
        Parameters
        ----------
        initial_state : str
            Name of the initial state. Must be a key of INITIAL_STATES.
        """
        self.set_initial_state(initial_state)

    # ------------------------------------------------------------------
    # State and operation sequence
    # ------------------------------------------------------------------
    def set_initial_state(self, name: str):
        """Resets the simulator with an initial state from INITIAL_STATES."""
        if name not in self.INITIAL_STATES:
            raise ValueError(f"Unknown state: {name}. "
                             f"Options: {list(self.INITIAL_STATES.keys())}")
        self._state = self.INITIAL_STATES[name].copy()
        self._trajectory: List[Tuple[float, float, float]] = []
        self._gate_labels: List[str] = []
        self._record()

    def reset(self, initial_state: str = "|0⟩"):
        """Semantic alias for set_initial_state."""
        self.set_initial_state(initial_state)

    def _record(self):
        """Saves the current Bloch vector position to the trajectory."""
        self._trajectory.append(QuantumMath.bloch_vector(self._state))

    @property
    def state(self) -> np.ndarray:
        """Current state vector of the qubit."""
        return self._state.copy()

    @property
    def bloch_vector(self) -> Tuple[float, float, float]:
        """Current Bloch vector (x, y, z)."""
        return QuantumMath.bloch_vector(self._state)

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------
    def apply_gate(self, gate: str, **kwargs):
        """Applies a gate to the current state and records the position.

        Parameters
        ----------
        gate : str
            Gate name. Fixed gates: "I", "X", "Y", "Z",
            "H", "S", "T", "Sdg", "Tdg". Parametric gates:
            "Rx", "Ry", "Rz" (require theta=<float>),
            "P" (requires phi=<float>).
        **kwargs
            Additional parameters for parametric gates:
            theta (rad) for Rx/Ry/Rz, phi (rad) for P.

        Raises
        ------
        ValueError
            If the gate name is not recognized.
        """
        if gate in self.GATES:
            U = self.GATES[gate]
            label = gate
        elif gate == "Rx":
            theta = kwargs.get("theta", np.pi)
            U = Gates.Rx(theta)
            label = f"Rx({theta:.2f})"
        elif gate == "Ry":
            theta = kwargs.get("theta", np.pi)
            U = Gates.Ry(theta)
            label = f"Ry({theta:.2f})"
        elif gate == "Rz":
            theta = kwargs.get("theta", np.pi)
            U = Gates.Rz(theta)
            label = f"Rz({theta:.2f})"
        elif gate == "P":
            phi = kwargs.get("phi", np.pi)
            U = Gates.P(phi)
            label = f"P({phi:.2f})"
        else:
            raise ValueError(f"Gate '{gate}' not recognized. "
                             f"Fixed: {list(self.GATES.keys())}. "
                             f"Parametric: Rx, Ry, Rz, P.")

        self._state = U @ self._state
        self._state = QuantumMath.normalize(self._state)
        self._gate_labels.append(label)
        self._record()

    def apply_sequence(self, sequence: List[dict]):
        """Applies a sequence of gates defined as a list of dicts.

        Each element of the list must be {"gate": <name>, **kwargs}.

        Example::

            sim.apply_sequence([
                {"gate": "H"},
                {"gate": "Rx", "theta": np.pi / 3},
                {"gate": "S"},
            ])
        """
        for step in sequence:
            gate = step.pop("gate")
            self.apply_gate(gate, **step)

    # ------------------------------------------------------------------
    # Plotly visualization (interactive)
    # ------------------------------------------------------------------
    def plot_trajectory(
        self,
        title: str = "Trajectory on the Bloch sphere",
        show_sphere: bool = True,
        dark_mode: bool = True,
    ) -> go.Figure:
        """Generates a Plotly 3D figure with the Bloch vector trajectory.

        Parameters
        ----------
        title : str
            Figure title.
        show_sphere : bool
            If True, draws the translucent sphere.
        dark_mode : bool
            If True, uses dark background in the style of the book.

        Returns
        -------
        go.Figure
            Plotly figure ready for .show() or st.plotly_chart().
        """
        fig = go.Figure()

        # --- Bloch sphere ---
        if show_sphere:
            u = np.linspace(0, 2 * np.pi, 60)
            v = np.linspace(0, np.pi, 40)
            sx = np.outer(np.cos(u), np.sin(v))
            sy = np.outer(np.sin(u), np.sin(v))
            sz = np.outer(np.ones_like(u), np.cos(v))
            fig.add_trace(go.Surface(
                x=sx, y=sy, z=sz,
                opacity=0.10,
                colorscale=[[0, "#58a6ff"], [1, "#58a6ff"]],
                showscale=False,
                hoverinfo="skip",
                name="Sphere",
            ))

        # --- Main axes ---
        axis_color = "#8b949e"
        for axis_end, label in [
            ((1.3, 0, 0), "X"), ((0, 1.3, 0), "Y"), ((0, 0, 1.3), "Z"),
        ]:
            fig.add_trace(go.Scatter3d(
                x=[0, axis_end[0]], y=[0, axis_end[1]], z=[0, axis_end[2]],
                mode="lines+text",
                line=dict(color=axis_color, width=2),
                text=["", label],
                textfont=dict(color=axis_color, size=14),
                showlegend=False,
                hoverinfo="skip",
            ))

        # Poles
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[1.15, -1.15],
            mode="text",
            text=["|0⟩", "|1⟩"],
            textfont=dict(color="#e6edf3", size=13),
            showlegend=False,
            hoverinfo="skip",
        ))

        # --- Trajectory ---
        xs = [p[0] for p in self._trajectory]
        ys = [p[1] for p in self._trajectory]
        zs = [p[2] for p in self._trajectory]

        labels = ["Start"] + self._gate_labels

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines+markers",
            line=dict(color="#f78166", width=4),
            marker=dict(
                size=[8 if i == 0 or i == len(xs)-1 else 5
                      for i in range(len(xs))],
                color="#f78166",
                opacity=0.9,
            ),
            text=labels,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                "<extra></extra>"
            ),
            name="Trajectory",
        ))

        # --- Final state vector ---
        x_f, y_f, z_f = xs[-1], ys[-1], zs[-1]
        fig.add_trace(go.Cone(
            x=[0], y=[0], z=[0],
            u=[x_f], v=[y_f], w=[z_f],
            colorscale=[[0, "#f78166"], [1, "#f78166"]],
            showscale=False,
            sizemode="absolute",
            sizeref=0.25,
            anchor="tail",
            name="Final state",
        ))

        # --- Layout ---
        bg = "#0d1117" if dark_mode else "white"
        text_col = "#e6edf3" if dark_mode else "#24292f"

        fig.update_layout(
            title=dict(text=title, font=dict(color=text_col, size=15)),
            paper_bgcolor=bg,
            scene=dict(
                bgcolor="#161b22" if dark_mode else "#f6f8fa",
                xaxis=dict(range=[-1.4, 1.4], showgrid=False,
                           zeroline=False, showticklabels=False),
                yaxis=dict(range=[-1.4, 1.4], showgrid=False,
                           zeroline=False, showticklabels=False),
                zaxis=dict(range=[-1.4, 1.4], showgrid=False,
                           zeroline=False, showticklabels=False),
                aspectmode="cube",
            ),
            legend=dict(
                font=dict(color=text_col),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    # ------------------------------------------------------------------
    # State information
    # ------------------------------------------------------------------
    def state_info(self) -> dict:
        """Returns a summary of the current qubit state.

        Returns
        -------
        dict
            Keys: alpha, beta, prob_0, prob_1, bloch_vector,
            theta_deg, phi_deg, purity.
        """
        alpha, beta = self._state[0], self._state[1]
        theta, phi = QuantumMath.bloch_angles(self._state)
        rho = QuantumMath.density_matrix(self._state)
        purity = float(np.real(np.trace(rho @ rho)))
        return {
            "alpha": alpha,
            "beta": beta,
            "prob_0": float(np.abs(alpha) ** 2),
            "prob_1": float(np.abs(beta) ** 2),
            "bloch_vector": QuantumMath.bloch_vector(self._state),
            "theta_deg": float(np.degrees(theta)),
            "phi_deg": float(np.degrees(phi)),
            "purity": purity,
        }
