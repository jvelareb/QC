# requirements: streamlit, numpy, matplotlib, qiskit, qiskit_aer, plotly, pylatexenc
import streamlit as st
import numpy as np
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.figure import Figure
import plotly.graph_objects as go
from matplotlib.lines import Line2D

import contextlib

# ---- Login
from auth import login_box

# ---- Qiskit (opcional)
QISKIT_AVAILABLE = True
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import Aer
    from qiskit.visualization import circuit_drawer, plot_histogram, plot_state_qsphere
except Exception:
    QISKIT_AVAILABLE = False

# ---------------- UI base ----------------
st.set_page_config(page_title="Quantum Toolkit", layout="wide", page_icon="⚛️")
st.title("⚛️ Quantum Toolkit — Esfera de Bloch, Puertas y Circuitos")

# ---------------- Login ------------------
role = login_box()
if role not in ("user", "admin"):
    st.info("Inicia sesión desde el panel lateral.")
    st.stop()

# ---------------- Utilidades 1 qubit ----------------
def ket_from_angles(theta_deg, phi_deg):
    th = np.deg2rad(theta_deg); ph = np.deg2rad(phi_deg)
    a = np.cos(th/2.0); b = np.exp(1j*ph)*np.sin(th/2.0)
    return np.array([[a],[b]], dtype=complex)

def ket_from_amplitudes(alpha, beta):
    v = np.array([[alpha],[beta]], complex)
    n = np.linalg.norm(v)
    return v if n < 1e-12 else (v/n)

def bloch_xyz(psi):
    rho = psi @ psi.conj().T
    sx = np.array([[0,1],[1,0]], complex)
    sy = np.array([[0,-1j],[1j,0]], complex)
    sz = np.array([[1,0],[0,-1]], complex)
    x = float(np.trace(rho @ sx).real)
    y = float(np.trace(rho @ sy).real)
    z = float(np.trace(rho @ sz).real)
    return x,y,z

def angles_from_xyz(x,y,z):
    theta = np.rad2deg(np.arccos(np.clip(z, -1.0, 1.0)))
    phi   = np.rad2deg(np.arctan2(y, x))
    return theta, phi

def U_gate(name, params=None):
    n = name.upper(); I = np.eye(2, dtype=complex)
    if n == "I":  return I
    if n == "X":  return np.array([[0,1],[1,0]], complex)
    if n == "Y":  return np.array([[0,-1j],[1j,0]], complex)
    if n == "Z":  return np.array([[1,0],[0,-1]], complex)
    if n == "H":  return (1/np.sqrt(2))*np.array([[1,1],[1,-1]], complex)
    if n == "S":  return np.array([[1,0],[0,1j]], complex)
    if n in ("S†","SDG"): return np.array([[1,0],[0,-1j]], complex)
    if n == "T":  return np.array([[1,0],[0,np.exp(1j*np.pi/4)]], complex)
    if n in ("T†","TDG"): return np.array([[1,0],[0,np.exp(-1j*np.pi/4)]], complex)
    if n == "RX":
        theta = float(params or 0.0)
        c, s = np.cos(theta/2), -1j*np.sin(theta/2)
        return np.array([[c, s],[s, c]], complex)
    if n == "RY":
        theta = float(params or 0.0)
        c, s = np.cos(theta/2), np.sin(theta/2)
        return np.array([[c, -s],[s, c]], complex)
    if n == "RZ":
        theta = float(params or 0.0)
        return np.array([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]], complex)
    if n in ("U","U3"):
        theta, phi, lam = params or (0.0,0.0,0.0)
        return np.array([
            [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
            [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]
        ], complex)
    return I

def fig_to_png_bytes(fig: Figure, dpi=800, *, bbox_inches="tight", pad_inches=0.02):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    buf.seek(0)
    return buf

# ---- input combinado (slider + número) sin warnings
def angle_input(label, key_base, min_val, max_val, *, step=0.1, unit="deg", default=0.0):
    slider_key = f"{key_base}_slider"; input_key = f"{key_base}_input"
    st.session_state.setdefault(slider_key, float(default))
    st.session_state.setdefault(input_key,  float(default))
    def _sync_from_slider(): st.session_state[input_key] = float(st.session_state[slider_key])
    def _sync_from_input():
        v = float(st.session_state[input_key])
        st.session_state[input_key]  = max(float(min_val), min(float(max_val), v))
        st.session_state[slider_key] = st.session_state[input_key]
    csl, cnum = st.columns([4, 1])
    csl.slider(f"{label} ({unit})", min_val, max_val, step=step, key=slider_key, on_change=_sync_from_slider)
    cnum.number_input(" ", min_value=float(min_val), max_value=float(max_val), step=float(step),
                      key=input_key, on_change=_sync_from_input, format="%.2f")
    return float(st.session_state[slider_key])

# ---- Bloch (Plotly)
DEFAULT_CAMERA = dict(eye=dict(x=1.6, y=1.6, z=1.1))
def bloch_plotly(states, colors, labels, title, camera_key, height=640):
    cam = st.session_state.get(camera_key, DEFAULT_CAMERA)
    fig = go.Figure()
    u, v = np.mgrid[0:2*np.pi:120j, 0:np.pi:120j]
    xs, ys, zs = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, showscale=False, opacity=0.12,
                             colorscale=[[0, '#8ab4f8'], [1, '#8ab4f8']], hoverinfo='skip'))
    L = 1.1
    for x,y,z in (([-L,L],[0,0],[0,0]), ([0,0],[-L,L],[0,0]), ([0,0],[0,0],[-L,L])):
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines",
                                   line=dict(color="black", width=4), hoverinfo='skip', showlegend=False))
    for (psi, col, lab) in zip(states, colors, labels):
        x,y,z = bloch_xyz(psi)
        fig.add_trace(go.Scatter3d(x=[0,x], y=[0,y], z=[0,z], mode="lines",
                                   line=dict(color=col, width=8), name=lab,
                                   hovertemplate=f"{lab}<br>x={x:.3f}<br>y={y:.3f}<br>z={z:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode="markers",
                                   marker=dict(color=col, size=6), showlegend=False))
    fig.update_layout(title=title, height=height, scene=dict(aspectmode='cube', camera=cam,
                        xaxis=dict(range=[-1.1,1.1]), yaxis=dict(range=[-1.1,1.1]), zaxis=dict(range=[-1.1,1.1])),
                      margin=dict(l=4,r=4,b=4,t=40), legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Estado 1-Qubit",
    "2. Puertas 1-Qubit",
    "3. Circuitos (Dashboard)",
    "4. Código Qiskit"
])

# ------ Tab 1
with tab1:
    st.subheader("Esfera de Bloch")
    left, right = st.columns([0.65, 1.35], gap="small")
    with left:
        method1 = st.radio("Método de entrada", ["Ángulos (θ, φ)", "Amplitudes (α, β)", "Presets"],
                           key="t1_method", horizontal=True)
        if method1 == "Ángulos (θ, φ)":
            th1 = angle_input("θ", "t1_theta", 0.0, 180.0, step=0.1, unit="deg", default=0.0)
            ph1 = angle_input("φ", "t1_phi",   0.0, 360.0, step=0.1, unit="deg", default=0.0)
            psi1 = ket_from_angles(th1, ph1)
        elif method1 == "Amplitudes (α, β)":
            a_re = st.number_input("α (real)", value=1.0, key="t1_are")
            a_im = st.number_input("α (imag)", value=0.0, key="t1_aim")
            b_re = st.number_input("β (real)", value=0.0, key="t1_bre")
            b_im = st.number_input("β (imag)", value=0.0, key="t1_bim")
            psi1 = ket_from_amplitudes(a_re + 1j*a_im, b_re + 1j*b_im)
        else:
            preset = st.selectbox("Preset", ["|0⟩","|1⟩","|+⟩","|−⟩","|+i⟩","|−i⟩"], key="t1_preset")
            mp = {"|0⟩":(1,0),"|1⟩":(0,1),"|+⟩":(1/np.sqrt(2),1/np.sqrt(2)),
                  "|−⟩":(1/np.sqrt(2),-1/np.sqrt(2)),"|+i⟩":(1/np.sqrt(2),1j/np.sqrt(2)),
                  "|−i⟩":(1/np.sqrt(2),-1j/np.sqrt(2))}
            alpha, beta = mp[preset]; psi1 = ket_from_amplitudes(alpha, beta)

        x,y,z = bloch_xyz(psi1); th, ph = angles_from_xyz(x,y,z)
        if st.button("🔄 Reset vista 3D", key="t1_reset"):
            st.session_state["t1_cam"] = DEFAULT_CAMERA
        fig_int = bloch_plotly([psi1], ['#d62728'], ['|ψ⟩'], "Esfera de Bloch", "t1_cam", height=760)
        st.plotly_chart(fig_int, use_container_width=True, key="t1_plotly")
        st.caption(f"Ángulos: θ={th:.2f}°, φ={ph:.2f}°  |  x={x:.3f}, y={y:.3f}, z={z:.3f}")

    with right:
        fig = Figure(figsize=(12,8)); ax = fig.add_subplot(111, projection="3d")
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
        xs, ys, zs = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
        ax.plot_surface(xs, ys, zs, color="#8ab4f8", alpha=0.12, edgecolor="#789", linewidth=0.25)
        x,y,z = bloch_xyz(psi1)
        ax.quiver(0,0,0,x,y,z, color="#d62728", arrow_length_ratio=0.08, length=1.0, linewidth=2.5)
        ax.scatter([x],[y],[z], color="#d62728", s=30)
        ax.set_axis_off(); ax.set_box_aspect((1,1,1)); ax.view_init(elev=22, azim=30)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇️ PNG (800dpi)", data=fig_to_png_bytes(fig, 800), file_name="bloch.png", mime="image/png")

# ------ Tab 2
with tab2:
    st.subheader("Puertas 1-Qubit: |ψ_in⟩ → |ψ_out⟩")
    top1, top2 = st.columns([1,1], gap="large")
    with top1:
        st.markdown("**Estado inicial**")
        method2 = st.radio("Método", ["Ángulos (θ, φ)", "Amplitudes (α, β)", "Presets"], key="t2_method", horizontal=True)
        if method2 == "Ángulos (θ, φ)":
            th2 = angle_input("θ_in", "t2_theta", 0.0, 180.0, step=0.1, unit="deg", default=0.0)
            ph2 = angle_input("φ_in", "t2_phi",   0.0, 360.0, step=0.1, unit="deg", default=0.0)
            psi_in = ket_from_angles(th2, ph2)
        elif method2 == "Amplitudes (α, β)":
            a_re2 = st.number_input("α (real)", value=1.0, key="t2_are")
            a_im2 = st.number_input("α (imag)", value=0.0, key="t2_aim")
            b_re2 = st.number_input("β (real)", value=0.0, key="t2_bre")
            b_im2 = st.number_input("β (imag)", value=0.0, key="t2_bim")
            psi_in = ket_from_amplitudes(a_re2 + 1j*a_im2, b_re2 + 1j*b_im2)
        else:
            preset2 = st.selectbox("Preset", ["|0⟩","|1⟩","|+⟩","|−⟩","|+i⟩","|−i⟩"], key="t2_preset")
            mp = {"|0⟩":(1,0),"|1⟩":(0,1),"|+⟩":(1/np.sqrt(2),1/np.sqrt(2)),
                  "|−⟩":(1/np.sqrt(2),-1/np.sqrt(2)),"|+i⟩":(1/np.sqrt(2),1j/np.sqrt(2)),
                  "|−i⟩":(1/np.sqrt(2),-1j/np.sqrt(2))}
            alpha, beta = mp[preset2]; psi_in = ket_from_amplitudes(alpha, beta)
    with top2:
        st.markdown("**Puerta**")
        gname = st.selectbox("Tipo", ["I","X","Y","Z","H","S","S†","T","T†","Rx","Ry","Rz","U(θ,φ,λ)"], key="t2_gate")
        if gname == "Rx":
            theta_rx = st.number_input("θ Rx (rad)", value=float(np.pi/2), key="t2_rx", step=0.01, format="%.3f")
            U = U_gate("RX", theta_rx); gate_desc = f"Rx({theta_rx:.3f} rad)"
        elif gname == "Ry":
            theta_ry = st.number_input("θ Ry (rad)", value=float(np.pi/2), key="t2_ry", step=0.01, format="%.3f")
            U = U_gate("RY", theta_ry); gate_desc = f"Ry({theta_ry:.3f} rad)"
        elif gname == "Rz":
            theta_rz = st.number_input("θ Rz (rad)", value=float(np.pi/2), key="t2_rz", step=0.01, format="%.3f")
            U = U_gate("RZ", theta_rz); gate_desc = f"Rz({theta_rz:.3f} rad)"
        elif gname == "U(θ,φ,λ)":
            t_u = angle_input("θ U", "t2_U_theta", -360.0, 360.0, step=0.1, unit="deg", default=90.0)
            p_u = angle_input("φ U", "t2_U_phi",   -360.0, 360.0, step=0.1, unit="deg", default=0.0)
            l_u = angle_input("λ U", "t2_U_lam",   -360.0, 360.0, step=0.1, unit="deg", default=0.0)
            U = U_gate("U", (np.deg2rad(t_u), np.deg2rad(p_u), np.deg2rad(l_u)))
            gate_desc = f"U(θ={t_u:.1f}°, φ={p_u:.1f}°, λ={l_u:.1f}°)"
        else:
            U = U_gate(gname); gate_desc = gname

    psi_out = U @ psi_in
    left, right = st.columns([0.7, 1.3], gap="small")
    with left:
        if st.button("🔄 Reset vista 3D", key="t2_reset"):
            st.session_state["t2_cam"] = DEFAULT_CAMERA
        fig2_int = bloch_plotly([psi_in, psi_out], ['#1f77b4','#d62728'],
                                ['|ψ_in⟩','|ψ_out⟩'], f"Esfera de Bloch — {gate_desc}", "t2_cam", height=760)
        st.plotly_chart(fig2_int, use_container_width=True, key="t2_plotly")
    with right:
        st.markdown("#### Resultado")
        st.code(str(psi_out), language="text")

# ------ Tab 3
with tab3:
    st.subheader("Circuitos predeterminados — Dashboard")
    if not QISKIT_AVAILABLE:
        st.error("Qiskit / qiskit_aer no están instalados.")
    else:
        left, right = st.columns([1,2], gap="large")
        with left:
            algo = st.selectbox("Algoritmo", [
                "Bell","GHZ","QFT","Deutsch-Jozsa","Bernstein-Vazirani","Grover"
            ], key="t3_algo")
            with st.form("t3_param_form"):
                if algo == "Bell":
                    measure = st.checkbox("Añadir medición", True)
                elif algo == "GHZ":
                    n = st.number_input("n (qubits ≥2)", min_value=2, value=3, step=1)
                    measure = st.checkbox("Añadir medición", True)
                elif algo == "QFT":
                    n = st.number_input("n (qubits ≥1)", min_value=1, value=3, step=1)
                    swaps = st.checkbox("SWAPs finales", True)
                    measure = st.checkbox("Añadir medición", True)
                elif algo == "Deutsch-Jozsa":
                    n = st.number_input("n (entrada ≥1)", min_value=1, value=3, step=1)
                    kind = st.selectbox("Oráculo", ["balanced","constant1"])
                elif algo == "Bernstein-Vazirani":
                    s = st.text_input("Secreto s (binario)", "1011")
                elif algo == "Grover":
                    n = st.number_input("n (qubits ≥2)", min_value=2, value=3, step=1)
                    marked = st.text_input("Marcado (binario longitud n)", "111")
                shots = st.slider("Shots", 100, 4096, 1024, step=100)
                submitted = st.form_submit_button("Generar y simular", use_container_width=True)

        if "t3_qc" not in st.session_state: st.session_state.t3_qc = None
        if "t3_counts" not in st.session_state: st.session_state.t3_counts = None
        if "t3_sv" not in st.session_state: st.session_state.t3_sv = None

        if submitted:
            try:
                if algo == "Bell":
                    qc = QuantumCircuit(2,2 if measure else 0); qc.h(0); qc.cx(0,1)
                    if measure: qc.measure([0,1],[0,1])
                elif algo == "GHZ":
                    qc = QuantumCircuit(int(n), int(n) if measure else 0); qc.h(0)
                    for i in range(int(n)-1): qc.cx(i,i+1)
                    if measure: qc.measure(range(int(n)), range(int(n)))
                elif algo == "QFT":
                    nn = int(n); qc = QuantumCircuit(nn, nn if measure else 0)
                    for j in range(nn):
                        qc.h(j)
                        for k in range(j+1, nn):
                            qc.cp(np.pi/2**(k-j), k, j)
                    if swaps:
                        for i in range(nn//2): qc.swap(i, nn-1-i)
                    if measure: qc.measure(range(nn), range(nn))
                elif algo == "Deutsch-Jozsa":
                    nn = int(n); oracle = QuantumCircuit(nn+1)
                    if kind=="balanced":
                        b = 1
                        sstr = format(b, f"0{nn}b")
                        for i,c in enumerate(sstr):
                            if c=="1": oracle.x(i)
                        for i in range(nn): oracle.cx(i,nn)
                        for i,c in enumerate(sstr):
                            if c=="1": oracle.x(i)
                    else:
                        oracle.x(nn)
                    qc = QuantumCircuit(nn+1, nn)
                    qc.x(nn); qc.h(nn); qc.h(range(nn))
                    qc.append(oracle.to_gate(label=f"Oracle({kind})"), range(nn+1))
                    qc.h(range(nn)); qc.measure(range(nn), range(nn))
                elif algo == "Bernstein-Vazirani":
                    if not all(c in "01" for c in s):
                        st.warning("El secreto s debe ser binario."); st.stop()
                    nn = len(s); qc = QuantumCircuit(nn+1, nn)
                    qc.x(nn); qc.h(range(nn+1)); qc.barrier()
                    for i,bit in enumerate(reversed(s)):
                        if bit=="1": qc.cx(i,nn)
                    qc.barrier(); qc.h(range(nn)); qc.measure(range(nn),range(nn))
                elif algo == "Grover":
                    if len(marked) != int(n) or not all(c in "01" for c in marked):
                        st.warning("Marcado debe ser binario de longitud n."); st.stop()
                    nn = int(n); qc = QuantumCircuit(nn, nn); qc.h(range(nn))
                    oracle = QuantumCircuit(nn, name="Oracle")
                    for i,c in enumerate(reversed(marked)):
                        if c=="0": oracle.x(i)
                    oracle.h(nn-1); oracle.mcx(list(range(nn-1)), nn-1); oracle.h(nn-1)
                    for i,c in enumerate(reversed(marked)):
                        if c=="0": oracle.x(i)
                    diffuser = QuantumCircuit(nn, name="Diffuser")
                    diffuser.h(range(nn)); diffuser.x(range(nn))
                    diffuser.h(nn-1); diffuser.mcx(list(range(nn-1)), nn-1); diffuser.h(nn-1)
                    diffuser.x(range(nn)); diffuser.h(range(nn))
                    iters = max(1, int(np.floor(np.pi/4*np.sqrt(2**nn))))
                    for _ in range(iters):
                        qc.append(oracle.to_gate(), range(nn))
                        qc.append(diffuser.to_gate(), range(nn))
                    qc.measure_all()

                st.session_state.t3_qc = qc

                # simulación
                sim = Aer.get_backend("aer_simulator")
                qcsv = qc.remove_final_measurements(inplace=False)
                qcsv.save_statevector()
                sv = sim.run(transpile(qcsv, sim)).result().get_statevector()
                st.session_state.t3_sv = sv

                counts = None
                if qc.num_clbits > 0:
                    qasm = Aer.get_backend("qasm_simulator")
                    counts = qasm.run(transpile(qc, qasm), shots=shots).result().get_counts()
                st.session_state.t3_counts = counts

            except Exception as e:
                st.error(f"Error al generar/simular: {e}")

        with right:
            dtabs = st.tabs(["Diagrama", "Medidas", "Q-sphere", "Statevector"])
            with dtabs[0]:
                if st.session_state.t3_qc is not None:
                    try:
                        figC = circuit_drawer(st.session_state.t3_qc, output="mpl", style={'name':'mpl'})
                        figC.set_size_inches(9, 3.2)
                        st.pyplot(figC, use_container_width=True)
                    except Exception as e:
                        st.error(f"No se pudo dibujar el circuito: {e}")
                else:
                    st.info("Genera el circuito a la izquierda.")
            with dtabs[1]:
                if st.session_state.t3_counts is not None:
                    figH = plot_histogram(st.session_state.t3_counts)
                    figH.set_size_inches(8.0, 3.0)
                    st.pyplot(figH, use_container_width=True)
                else:
                    st.info("Este circuito no tiene mediciones o no se ha simulado.")
            with dtabs[2]:
                if st.session_state.t3_sv is not None:
                    figQ = plot_state_qsphere(Statevector(st.session_state.t3_sv))
                    figQ.set_size_inches(6.0, 6.0)
                    st.pyplot(figQ, use_container_width=False)
                else:
                    st.info("Vector de estado no disponible.")
            with dtabs[3]:
                if st.session_state.t3_sv is not None:
                    st.code(str(st.session_state.t3_sv), language="text")
                else:
                    st.info("Vector de estado no disponible.")

# ------ Tab 4
with tab4:
    st.subheader("Ejecutor de código Qiskit (define una variable `qc`)")
    if not QISKIT_AVAILABLE:
        st.error("Qiskit / qiskit_aer no están instalados.")
    else:
        code = st.text_area("Código Python", height=240, key="t4_code", value=
"""from qiskit import QuantumCircuit
# Ejemplo: GHZ de 3 qubits
qc = QuantumCircuit(3,3)
qc.h(0); qc.cx(0,1); qc.cx(0,2); qc.measure_all()
""")
        if st.button("Ejecutar y simular", key="t4_run", type="primary"):
            ns = {}
            try:
                exec(code, {"np":np, "QuantumCircuit":QuantumCircuit}, ns)
                qc = ns.get("qc", None)
                if qc is None:
                    st.warning("Tu código no definió `qc`."); st.stop()
                figC = circuit_drawer(qc, output="mpl", style={'name':'mpl'})
                figC.set_size_inches(9, 3.2)
                st.pyplot(figC, use_container_width=True)
                sim = Aer.get_backend("aer_simulator")
                qcsv = qc.remove_final_measurements(inplace=False); qcsv.save_statevector()
                sv = sim.run(transpile(qcsv, sim)).result().get_statevector()
                st.markdown("**Vector de estado (sin medidas)**")
                st.code(str(sv), language="text")
                if qc.num_clbits > 0:
                    qasm = Aer.get_backend("qasm_simulator")
                    counts = qasm.run(transpile(qc, qasm), shots=1024).result().get_counts()
                    figH = plot_histogram(counts)
                    figH.set_size_inches(8.0, 3.0)
                    st.pyplot(figH, use_container_width=True)
            except Exception as e:
                st.error(f"Error al ejecutar tu código: {e}")
