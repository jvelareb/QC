import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import streamlit as st

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_histogram, circuit_drawer
from qiskit_aer import Aer

st.title("⚛️ QC — Qiskit Minimal (Streamlit)")

@st.cache_resource(show_spinner=False)
def get_backends():
    return {
        "aer": Aer.get_backend("aer_simulator"),
        "qasm": Aer.get_backend("qasm_simulator"),
    }

@st.cache_data(show_spinner=False)
def fig_to_png(fig: Figure, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    buf.seek(0); return buf

def ket_from_angles(theta_deg: float, phi_deg: float):
    th = np.deg2rad(theta_deg); ph = np.deg2rad(phi_deg)
    a = np.cos(th/2.0); b = np.exp(1j*ph)*np.sin(th/2.0)
    v = np.array([a, b], dtype=complex)
    return v / np.linalg.norm(v)

tab1, tab2 = st.tabs(["1) Bloch 1-Qubit", "2) Circuitos + Simulación"])

# ---------- TAB 1: Bloch ----------
with tab1:
    st.subheader("Esfera de Bloch (Qiskit + Matplotlib)")
    c1, c2 = st.columns(2)
    with c1:
        th = st.slider("θ (deg)", 0.0, 180.0, 45.0, 0.1)
        ph = st.slider("φ (deg)", 0.0, 360.0, 30.0, 0.1)
    state = ket_from_angles(th, ph)
    sv = Statevector(state)

    try:
        fig = plot_bloch_multivector(sv)
        fig.set_size_inches(5.0, 5.0)
        st.pyplot(fig, use_container_width=False)
        st.download_button("⬇️ PNG (Bloch)", data=fig_to_png(fig), file_name="bloch.png", mime="image/png")
        plt.close(fig)
    except Exception as e:
        st.error("No se pudo renderizar la Bloch.")
        st.exception(e)

# ---------- TAB 2: Circuitos ----------
with tab2:
    st.subheader("Circuitos predeterminados (Aer)")
    left, right = st.columns([1,2], gap="large")
    with left:
        algo = st.selectbox("Algoritmo", ["Bell", "GHZ (n)", "QFT (n)", "Grover demo (n=3)"])
        if algo in ("GHZ (n)", "QFT (n)"):
            n = st.number_input("n (qubits)", min_value=2, value=3, step=1)
        shots = st.slider("Shots", 100, 4096, 1024, step=100)
        run = st.button("Generar + simular", type="primary", use_container_width=True)

    if run:
        try:
            # Construcción
            if algo == "Bell":
                qc = QuantumCircuit(2,2); qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1])
            elif algo == "GHZ (n)":
                qc = QuantumCircuit(n, n); qc.h(0)
                for i in range(n-1): qc.cx(i, i+1)
                qc.measure(range(n), range(n))
            elif algo == "QFT (n)":
                qc = QuantumCircuit(n, n)
                for j in range(n):
                    qc.h(j)
                    for k in range(j+1, n):
                        qc.cp(np.pi/2**(k-j), k, j)
                for i in range(n//2): qc.swap(i, n-1-i)
                qc.measure(range(n), range(n))
            else:  # Grover demo n=3 marcado "111"
                n = 3
                qc = QuantumCircuit(n, n); qc.h(range(n))
                oracle = QuantumCircuit(n)
                oracle.h(n-1); oracle.mcx(list(range(n-1)), n-1); oracle.h(n-1)
                diffuser = QuantumCircuit(n)
                diffuser.h(range(n)); diffuser.x(range(n))
                diffuser.h(n-1); diffuser.mcx(list(range(n-1)), n-1); diffuser.h(n-1)
                diffuser.x(range(n)); diffuser.h(range(n))
                qc.append(oracle.to_gate(label="Oracle"), range(n))
                qc.append(diffuser.to_gate(label="Diff"), range(n))
                qc.measure_all()

            # Diagrama
            try:
                figC = circuit_drawer(qc, output="mpl", style={'name':'mpl'})
                figC.set_size_inches(8.5, 2.8)
                st.pyplot(figC, use_container_width=True)
                st.download_button("⬇️ Diagrama (PNG)", data=fig_to_png(figC), file_name="circuit.png", mime="image/png")
                plt.close(figC)
            except Exception as e:
                st.warning("No se pudo dibujar el circuito.")
                st.exception(e)

            # Simulación
            b = get_backends()
            qasm = b["qasm"]
            res = qasm.run(transpile(qc, qasm), shots=int(shots)).result()
            counts = res.get_counts()

            figH = plot_histogram(counts)
            figH.set_size_inches(7.0, 3.0)
            st.pyplot(figH, use_container_width=True)
            st.download_button("⬇️ Histograma (PNG)", data=fig_to_png(figH), file_name="hist.png", mime="image/png")
            plt.close(figH)

        except Exception as e:
            st.error("Error al generar/simular.")
            st.exception(e)

st.caption("Proyecto minimal con Qiskit + Aer, sin Plotly. Listo para Railway.")
