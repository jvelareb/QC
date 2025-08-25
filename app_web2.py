# app_web2.py
# Requisitos: streamlit, numpy, matplotlib, qiskit, qiskit-aer, pylatexenc
import io
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # backend headless
import matplotlib.pyplot as plt

from auth import login

# --- Config de página (solo aquí) ---
st.set_page_config(page_title="Quantum Toolkit", layout="wide", page_icon="⚛️")

# --- Login ---
role, api_key = login()
if not role:
    st.stop()
st.sidebar.success(f"Sesión: **{role}**")

# --- Intento de importar Qiskit (opcional) ---
QISKIT_AVAILABLE = True
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import Aer
    from qiskit.visualization import circuit_drawer, plot_histogram
except Exception:
    QISKIT_AVAILABLE = False

# ================= Utilidades Bloch =================
def ket_from_angles(theta_deg: float, phi_deg: float):
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    a = np.cos(th/2.0)
    b = np.exp(1j*ph)*np.sin(th/2.0)
    v = np.array([a, b], dtype=complex)
    return v / np.linalg.norm(v)

def bloch_xyz(psi: np.ndarray):
    """psi: vector de 2 complejos (a,b)."""
    a, b = psi
    # Pauli expectation values
    x = 2*np.real(np.conj(a)*b)
    y = 2*np.imag(np.conj(b)*a)
    z = np.abs(a)**2 - np.abs(b)**2
    return float(x), float(y), float(z)

def draw_bloch_matplotlib(theta_deg: float, phi_deg: float):
    psi = ket_from_angles(theta_deg, phi_deg)
    x, y, z = bloch_xyz(psi)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # esfera
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="#8ab4f8", alpha=0.15, linewidth=0, antialiased=True)

    # ejes
    L = 1.1
    ax.plot([-L, L], [0, 0], [0, 0], 'k-', lw=1)
    ax.plot([0, 0], [-L, L], [0, 0], 'k-', lw=1)
    ax.plot([0, 0], [0, 0], [-L, L], 'k-', lw=1)

    # vector estado
    ax.quiver(0, 0, 0, x, y, z, color="#d62728", arrow_length_ratio=0.07, linewidth=2)
    ax.scatter([x], [y], [z], color="#d62728", s=40)

    # etiquetas base
    ax.text(0, 0, 1.2, "|0⟩", color="navy", ha="center")
    ax.text(0, 0, -1.2, "|1⟩", color="navy", ha="center")
    ax.text(1.2, 0, 0, "|+⟩", color="navy", ha="center")
    ax.text(-1.2, 0, 0, "|-⟩", color="navy", ha="center")
    ax.text(0, 1.2, 0, "|+i⟩", color="navy", ha="center")
    ax.text(0, -1.2, 0, "|-i⟩", color="navy", ha="center")

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
    ax.axis("off")
    ax.view_init(elev=22, azim=30)
    return fig, (x, y, z)

def fig_to_png_bytes(fig, dpi=300):
    """Sin @st.cache_data para evitar UnhashableParamError."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    buf.seek(0)
    return buf

# ================= UI =================
st.title("⚛️ Quantum Toolkit — App con login")

tab1, tab2 = st.tabs(["1) Esfera de Bloch", "2) Circuito Bell (Qiskit)"])

with tab1:
    st.subheader("Esfera de Bloch (Matplotlib, estable en servidor)")
    # Controles (evita warning de Streamlit usando session_state como fuente de verdad)
    c1, c2 = st.columns(2)
    if "theta" not in st.session_state: st.session_state["theta"] = 0.0
    if "phi" not in st.session_state: st.session_state["phi"] = 0.0

    theta = c1.slider("θ (grados)", 0.0, 180.0, key="theta")
    phi   = c2.slider("φ (grados)", 0.0, 360.0, key="phi")

    fig, coords = draw_bloch_matplotlib(theta, phi)
    st.pyplot(fig, use_container_width=False)
    st.caption(f"Coordenadas Bloch: x={coords[0]:.3f}, y={coords[1]:.3f}, z={coords[2]:.3f}")
    st.download_button("⬇️ Descargar PNG", data=fig_to_png_bytes(fig, dpi=600),
                       file_name="bloch.png", mime="image/png")
    plt.close(fig)  # 🔒 importante en servidores

with tab2:
    st.subheader("Circuito Bell (si Qiskit está disponible)")
    if not QISKIT_AVAILABLE:
        st.error("Qiskit / qiskit-aer no están instalados o falló la importación.")
    else:
        measure = st.checkbox("Añadir medición", True)
        shots   = st.slider("Shots", 100, 4096, 1024, step=100)

        # Construcción del circuito Bell
        qreg = 2
        qc = QuantumCircuit(qreg, qreg if measure else 0)
        qc.h(0); qc.cx(0, 1)
        if measure:
            qc.measure([0, 1], [0, 1])

        # Dibujo del circuito
        try:
            figC = circuit_drawer(qc, output="mpl", style={'name':'mpl'})
            st.pyplot(figC, use_container_width=True)
            plt.close(figC)
        except Exception as e:
            st.warning(f"No se pudo dibujar el circuito (mpl): {e}")

        # Simulación
        try:
            if measure:
                backend = Aer.get_backend("qasm_simulator")
                result = backend.run(transpile(qc, backend), shots=shots).result()
                counts = result.get_counts()
                st.write("Resultados (counts):", counts)
                # Histograma
                try:
                    figH = plot_histogram(counts)
                    st.pyplot(figH, use_container_width=True)
                    plt.close(figH)
                except Exception as e:
                    st.warning(f"No se pudo dibujar el histograma: {e}")
            else:
                sim = Aer.get_backend("aer_simulator")
                qcsv = qc.remove_final_measurements(inplace=False)
                qcsv.save_statevector()
                sv = sim.run(transpile(qcsv, sim)).result().get_statevector()
                st.code(str(sv), language="text")
        except Exception as e:
            st.error(f"Error al simular: {e}")

st.success("App cargada correctamente.")
