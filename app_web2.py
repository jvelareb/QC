# app_web2.py — 4 pestañas: (1) Estado 1-Qubit, (2) Puertas, (3) Dashboard circuitos, (4) Código Qiskit
# Requisitos clave: streamlit, numpy, matplotlib, qiskit, qiskit-aer, pylatexenc

import io
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # backend sin GUI (servidor)
import matplotlib.pyplot as plt

from auth import login

# --- Config de página (solo aquí) ---
st.set_page_config(page_title="Quantum Toolkit", layout="wide", page_icon="⚛️")

# --- Login ---
role, api_key = login()
if not role:
    st.stop()
st.sidebar.success(f"Sesión: **{role}**")

# --- Qiskit opcional (pestañas 3 y 4) ---
QISKIT_AVAILABLE = True
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import Aer
    from qiskit.visualization import circuit_drawer, plot_histogram
except Exception:
    QISKIT_AVAILABLE = False

# ================= Utilidades de estado (1 qubit) =================
def ket_from_angles(theta_deg: float, phi_deg: float):
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    a = np.cos(th/2.0)
    b = np.exp(1j*ph)*np.sin(th/2.0)
    v = np.array([a, b], dtype=complex)
    n = np.linalg.norm(v)
    if n == 0:
        return np.array([1.0+0j, 0.0+0j])
    return v / n

def ket_from_amplitudes(alpha: complex, beta: complex):
    v = np.array([alpha, beta], dtype=complex)
    n = np.linalg.norm(v)
    if n == 0:
        return np.array([1.0+0j, 0.0+0j])
    return v / n

def bloch_xyz(psi: np.ndarray):
    a, b = psi
    x = 2*np.real(np.conj(a)*b)
    y = 2*np.imag(np.conj(b)*a)
    z = np.abs(a)**2 - np.abs(b)**2
    return float(x), float(y), float(z)

def angles_from_xyz(x,y,z):
    theta = float(np.rad2deg(np.arccos(np.clip(z, -1.0, 1.0))))
    phi   = float(np.rad2deg(np.arctan2(y, x)))
    return theta, phi

def fig_to_png_bytes(fig, dpi=600):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    buf.seek(0)
    return buf

# ---- Matrices de puertas 1-qubit ----
def U_gate(name, params=None):
    n = name.upper()
    I = np.eye(2, dtype=complex)
    if n == "I": return I
    if n == "X": return np.array([[0,1],[1,0]], complex)
    if n == "Y": return np.array([[0,-1j],[1j,0]], complex)
    if n == "Z": return np.array([[1,0],[0,-1]], complex)
    if n == "H": return (1/np.sqrt(2))*np.array([[1,1],[1,-1]], complex)
    if n == "S": return np.array([[1,0],[0,1j]], complex)
    if n in ("S†","SDG"): return np.array([[1,0],[0,-1j]], complex)
    if n == "T": return np.array([[1,0],[0,np.exp(1j*np.pi/4)]], complex)
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

def pretty_c(z, d=3):
    return f"{z.real:.{d}f}{z.imag:+.{d}f}j"

# ---- Dibujo de Bloch (Matplotlib) ----
def draw_bloch(theta_deg: float, phi_deg: float, color="#d62728", title=None):
    psi = ket_from_angles(theta_deg, phi_deg)
    x, y, z = bloch_xyz(psi)

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection='3d')

    # esfera
    u = np.linspace(0, 2*np.pi, 120)
    v = np.linspace(0, np.pi, 120)
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
    ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.07, linewidth=2)
    ax.scatter([x], [y], [z], color=color, s=40)

    # etiquetas base
    ax.text(0, 0, 1.2, "|0⟩", color="navy", ha="center")
    ax.text(0, 0, -1.2, "|1⟩", color="navy", ha="center")
    ax.text(1.2, 0, 0, "|+⟩", color="navy", ha="center")
    ax.text(-1.2, 0, 0, "|-⟩", color="navy", ha="center")
    ax.text(0, 1.2, 0, "|+i⟩", color="navy", ha="center")
    ax.text(0, -1.2, 0, "|-i⟩", color="navy", ha="center")

    ax.set_box_aspect((1,1,1))
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
    ax.axis("off")
    ax.view_init(elev=22, azim=30)
    if title:
        ax.set_title(title, pad=6)
    return fig, (x,y,z), psi

# ========================= UI (4 pestañas) =========================
st.title("⚛️ Quantum Toolkit — 4 pestañas")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Estado 1-Qubit",
    "2. Puertas 1-Qubit",
    "3. Circuitos (Dashboard)",
    "4. Código Qiskit"
])

# ---------------- Pestaña 1: Estado 1-Qubit ----------------
with tab1:
    st.subheader("Esfera de Bloch (estado de 1 qubit)")
    c1, c2, c3 = st.columns([1,1,2])

    method = c1.radio("Entrada", ["Ángulos (θ, φ)", "Amplitudes (α, β)", "Presets"], horizontal=False)

    if method == "Ángulos (θ, φ)":
        if "t1_theta" not in st.session_state: st.session_state["t1_theta"] = 0.0
        if "t1_phi" not in st.session_state:   st.session_state["t1_phi"]   = 0.0
        theta = c2.slider("θ (°)", 0.0, 180.0, key="t1_theta")
        phi   = c2.slider("φ (°)", 0.0, 360.0, key="t1_phi")
        fig1, coords, psi = draw_bloch(theta, phi, color="#d62728", title="Estado |ψ⟩")
    elif method == "Amplitudes (α, β)":
        a_re = c2.number_input("α (real)", value=1.0)
        a_im = c2.number_input("α (imag)", value=0.0)
        b_re = c2.number_input("β (real)", value=0.0)
        b_im = c2.number_input("β (imag)", value=0.0)
        psi = ket_from_amplitudes(a_re + 1j*a_im, b_re + 1j*b_im)
        x,y,z = bloch_xyz(psi)
        th, ph = angles_from_xyz(x,y,z)
        fig1, coords, _ = draw_bloch(th, ph, color="#d62728", title="Estado |ψ⟩")
    else:
        preset = c2.selectbox("Preset", ["|0⟩","|1⟩","|+⟩","|−⟩","|+i⟩","|−i⟩"])
        mp = {"|0⟩":(1,0),"|1⟩":(0,1),"|+⟩":(1/np.sqrt(2),1/np.sqrt(2)),
              "|−⟩":(1/np.sqrt(2),-1/np.sqrt(2)),"|+i⟩":(1/np.sqrt(2),1j/np.sqrt(2)),
              "|−i⟩":(1/np.sqrt(2),-1j/np.sqrt(2))}
        a,b = mp[preset]
        psi = ket_from_amplitudes(a,b)
        x,y,z = bloch_xyz(psi)
        th, ph = angles_from_xyz(x,y,z)
        fig1, coords, _ = draw_bloch(th, ph, color="#d62728", title=f"Estado {preset}")

    st.pyplot(fig1, use_container_width=False); plt.close(fig1)
    x,y,z = coords
    th, ph = angles_from_xyz(x,y,z)
    p0, p1 = abs(psi[0])**2*100, abs(psi[1])**2*100
    st.caption(f"Ángulos: θ={th:.2f}°, φ={ph:.2f}° | Coords: x={x:.3f}, y={y:.3f}, z={z:.3f} | P(0)={p0:.1f}%, P(1)={p1:.1f}%")
    st.download_button("⬇️ PNG (Bloch)", data=fig_to_png_bytes(fig_to_png:=plt.figure()) if False else fig_to_png_bytes(plt.figure()),
                       file_name="bloch.png", mime="image/png", disabled=True, help="(Usa el botón de guardar de tu navegador)")

# ---------------- Pestaña 2: Puertas 1-Qubit ----------------
with tab2:
    st.subheader("Aplicar puerta 1-qubit: |ψ_in⟩ → |ψ_out⟩")
    left, right = st.columns([1,1])

    with left:
        st.markdown("**Estado inicial**")
        m2 = st.radio("Entrada", ["Ángulos (θ, φ)", "Amplitudes (α, β)", "Presets"], horizontal=True)
        if m2 == "Ángulos (θ, φ)":
            if "t2_theta" not in st.session_state: st.session_state["t2_theta"] = 0.0
            if "t2_phi" not in st.session_state:   st.session_state["t2_phi"]   = 0.0
            th2 = st.slider("θ_in (°)", 0.0, 180.0, key="t2_theta")
            ph2 = st.slider("φ_in (°)", 0.0, 360.0, key="t2_phi")
            psi_in = ket_from_angles(th2, ph2)
        elif m2 == "Amplitudes (α, β)":
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
            a,b = mp[preset2]
            psi_in = ket_from_amplitudes(a,b)

    with right:
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
            t_u = st.number_input("θ U (°)", value=90.0, key="t2_U_theta")
            p_u = st.number_input("φ U (°)", value=0.0, key="t2_U_phi")
            l_u = st.number_input("λ U (°)", value=0.0, key="t2_U_lam")
            U = U_gate("U", (np.deg2rad(t_u), np.deg2rad(p_u), np.deg2rad(l_u)))
            gate_desc = f"U(θ={t_u:.1f}°, φ={p_u:.1f}°, λ={l_u:.1f}°)"
        else:
            U = U_gate(gname); gate_desc = gname

    psi_out = U @ psi_in
    xin,yin,zin = bloch_xyz(psi_in)
    xout,yout,zout = bloch_xyz(psi_out)

    # Dibujo IN/OUT
    fig_in, _, _  = draw_bloch(*angles_from_xyz(xin,yin,zin), color="#1f77b4", title="|ψ_in⟩")
    fig_out, _, _ = draw_bloch(*angles_from_xyz(xout,yout,zout), color="#d62728", title=f"|ψ_out⟩ = {gate_desc}")
    col_in, col_out = st.columns(2)
    with col_in:  st.pyplot(fig_in,  use_container_width=False); plt.close(fig_in)
    with col_out: st.pyplot(fig_out, use_container_width=False); plt.close(fig_out)

    st.markdown("#### Resultado matemático")
    st.code(
        f"U = [[{pretty_c(U[0,0])}, {pretty_c(U[0,1])}],\n"
        f"     [{pretty_c(U[1,0])}, {pretty_c(U[1,1])}]]\n\n"
        f"|ψ_in⟩ = [{pretty_c(psi_in[0])}, {pretty_c(psi_in[1])}]^T\n"
        f"|ψ_out⟩ = U · |ψ_in⟩ = [{pretty_c(psi_out[0])}, {pretty_c(psi_out[1])}]^T",
        language="text"
    )

# ---------------- Pestaña 3: Dashboard de circuitos ----------------
with tab3:
    st.subheader("Circuitos predeterminados — Dashboard")
    if not QISKIT_AVAILABLE:
        st.error("Qiskit / qiskit_aer no están instalados.")
    else:
        left, right = st.columns([1,2], gap="large")

        with left:
            algo = st.selectbox("Algoritmo", [
                "Bell", "GHZ", "QFT", "Deutsch-Jozsa",
                "Bernstein-Vazirani", "Grover", "Shor (demo N=15)"
            ])

            with st.form("t3_form"):
                measure = True
                params = {}
                if algo == "Bell":
                    measure = st.checkbox("Añadir medición", True)
                elif algo == "GHZ":
                    params["n"] = st.number_input("n (qubits ≥2)", min_value=2, value=3, step=1)
                    measure = st.checkbox("Añadir medición", True)
                elif algo == "QFT":
                    params["n"] = st.number_input("n (qubits ≥1)", min_value=1, value=3, step=1)
                    params["swaps"] = st.checkbox("SWAPs finales", True)
                    measure = st.checkbox("Añadir medición", True)
                elif algo == "Deutsch-Jozsa":
                    params["n"] = st.number_input("n (entrada ≥1)", min_value=1, value=3, step=1)
                    params["kind"] = st.selectbox("Oráculo", ["balanced","constant1"])
                elif algo == "Bernstein-Vazirani":
                    params["s"] = st.text_input("Secreto s (binario)", "1011")
                elif algo == "Grover":
                    params["n"] = st.number_input("n (qubits ≥2)", min_value=2, value=3, step=1)
                    params["marked"] = st.text_input("Marcado (binario longitud n)", "111")
                elif algo == "Shor (demo N=15)":
                    params["a"] = st.selectbox("a (coprimo con 15)", [2,7,8,11,13])

                shots = st.slider("Shots", 100, 8192, 1024, step=100)
                submitted = st.form_submit_button("Generar y simular", use_container_width=True)

        def qc_bell(measure=True):
            qc = QuantumCircuit(2, 2 if measure else 0)
            qc.h(0); qc.cx(0,1)
            if measure: qc.measure([0,1],[0,1])
            return qc

        def qc_ghz(n=3, measure=True):
            qc = QuantumCircuit(n, n if measure else 0)
            qc.h(0)
            for i in range(n-1): qc.cx(i, i+1)
            if measure: qc.measure(range(n), range(n))
            return qc

        def qc_qft(n=3, swaps=True, measure=True):
            qc = QuantumCircuit(n, n if measure else 0)
            for j in range(n):
                qc.h(j)
                for k in range(j+1, n):
                    qc.cp(np.pi/2**(k-j), k, j)
            if swaps:
                for i in range(n//2): qc.swap(i, n-1-i)
            if measure: qc.measure(range(n), range(n))
            return qc

        def qc_deutsch_jozsa(n=3, kind="balanced"):
            oracle = QuantumCircuit(n+1)
            if kind=="balanced":
                b = 1
                s = format(b, f"0{n}b")
                for i,c in enumerate(s):
                    if c=="1": oracle.x(i)
                for i in range(n): oracle.cx(i,n)
                for i,c in enumerate(s):
                    if c=="1": oracle.x(i)
            elif kind=="constant1":
                oracle.x(n)
            qc = QuantumCircuit(n+1, n)
            qc.x(n); qc.h(n); qc.h(range(n))
            qc.append(oracle.to_gate(label=f"Oracle({kind})"), range(n+1))
            qc.h(range(n)); qc.measure(range(n), range(n))
            return qc

        def qc_bernstein_vazirani(s="1011"):
            n = len(s); qc = QuantumCircuit(n+1, n)
            qc.x(n); qc.h(range(n+1)); qc.barrier()
            for i,bit in enumerate(reversed(s)):
                if bit=="1": qc.cx(i,n)
            qc.barrier(); qc.h(range(n)); qc.measure(range(n),range(n))
            return qc

        def _mcx(qc, ctrls, tgt):
            try:
                qc.mcx(ctrls, tgt)
            except Exception:
                raise RuntimeError("Tu Qiskit no soporta 'mcx'. Usa menos controles o actualiza.")

        def qc_grover(n=3, marked=None):
            if marked is None: marked = "1"*n
            if len(marked) != n or any(c not in "01" for c in marked):
                raise ValueError("marked debe ser binario de longitud n")
            qc = QuantumCircuit(n, n); qc.h(range(n))
            oracle = QuantumCircuit(n, name="Oracle")
            for i,c in enumerate(reversed(marked)):
                if c=="0": oracle.x(i)
            oracle.h(n-1); _mcx(oracle, list(range(n-1)), n-1); oracle.h(n-1)
            for i,c in enumerate(reversed(marked)):
                if c=="0": oracle.x(i)

            diffuser = QuantumCircuit(n, name="Diffuser")
            diffuser.h(range(n)); diffuser.x(range(n))
            diffuser.h(n-1); _mcx(diffuser, list(range(n-1)), n-1); diffuser.h(n-1)
            diffuser.x(range(n)); diffuser.h(range(n))

            iters = max(1, int(np.floor(np.pi/4*np.sqrt(2**n))))
            for _ in range(iters):
                qc.append(oracle.to_gate(), range(n))
                qc.append(diffuser.to_gate(), range(n))
            qc.measure_all()
            return qc

        def qc_shor_demo_15(a=2):
            def c_amod15(a, power):
                U = QuantumCircuit(4)
                for _ in range(power):
                    if a in [2,13]: U.swap(0,1); U.swap(1,2); U.swap(2,3)
                    if a in [7,8]: U.swap(2,3); U.swap(1,2); U.swap(0,1)
                    if a == 11: U.swap(1,3); U.swap(0,2)
                    if a in [7,11,13]:
                        for q in range(4): U.x(q)
                return U.to_gate(label=f"{a}^k mod 15").control()
            n_count = 8
            qc = QuantumCircuit(n_count+4, n_count)
            for q in range(n_count): qc.h(q)
            qc.x(n_count+3)
            for q in range(n_count):
                qc.append(c_amod15(a, 2**q), [q] + list(range(n_count, n_count+4)))
            # QFT^-1
            for j in range(n_count//2): qc.swap(j, n_count-1-j)
            for j in range(n_count):
                for m in range(j):
                    qc.cp(-np.pi/2**(j-m), m, j)
                qc.h(j)
            qc.measure(range(n_count), range(n_count))
            return qc

        # Estados en sesión
        if "t3_qc" not in st.session_state: st.session_state.t3_qc = None
        if "t3_counts" not in st.session_state: st.session_state.t3_counts = None
        if "t3_sv" not in st.session_state: st.session_state.t3_sv = None

        if submitted:
            try:
                if algo == "Bell":
                    qc = qc_bell(measure)
                elif algo == "GHZ":
                    qc = qc_ghz(int(params["n"]), measure)
                elif algo == "QFT":
                    qc = qc_qft(int(params["n"]), bool(params["swaps"]), measure)
                elif algo == "Deutsch-Jozsa":
                    qc = qc_deutsch_jozsa(int(params["n"]), params["kind"])
                elif algo == "Bernstein-Vazirani":
                    s = params["s"]
                    if not all(c in "01" for c in s):
                        st.warning("El secreto s debe ser binario."); st.stop()
                    qc = qc_bernstein_vazirani(s)
                elif algo == "Grover":
                    n = int(params["n"]); marked = params["marked"]
                    if len(marked) != n or not all(c in "01" for c in marked):
                        st.warning("Marcado debe ser binario de longitud n."); st.stop()
                    qc = qc_grover(n, marked)
                elif algo == "Shor (demo N=15)":
                    qc = qc_shor_demo_15(int(params["a"]))

                st.session_state.t3_qc = qc

                # Simulación
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
            dtabs = st.tabs(["Diagrama", "Medidas", "Statevector"])

            with dtabs[0]:
                if st.session_state.t3_qc is not None:
                    try:
                        figC = circuit_drawer(st.session_state.t3_qc, output="mpl", style={'name':'mpl'})
                        st.pyplot(figC, use_container_width=True)
                        plt.close(figC)
                    except Exception as e:
                        st.error(f"No se pudo dibujar el circuito: {e}")
                else:
                    st.info("Genera el circuito a la izquierda.")

            with dtabs[1]:
                if st.session_state.t3_counts is not None:
                    try:
                        figH = plot_histogram(st.session_state.t3_counts)
                        st.pyplot(figH, use_container_width=True)
                        plt.close(figH)
                    except Exception as e:
                        st.error(f"No se pudo dibujar el histograma: {e}")
                else:
                    st.info("Este circuito no tiene mediciones o no se ha simulado.")

            with dtabs[2]:
                if st.session_state.t3_sv is not None:
                    st.code(str(st.session_state.t3_sv), language="text")
                else:
                    st.info("Vector de estado no disponible.")

# ---------------- Pestaña 4: Código Qiskit ----------------
with tab4:
    st.subheader("Ejecutor de código Qiskit (define una variable `qc`)")
    if not QISKIT_AVAILABLE:
        st.error("Qiskit / qiskit_aer no están instalados.")
    else:
        code = st.text_area("Código Python", height=240, key="t4_code", value=
"""from qiskit import QuantumCircuit
# Ejemplo: GHZ de 3 qubits
qc = QuantumCircuit(3,3)
qc.h(0); qc.cx(0,1); qc.cx(0,2); qc.measure(range(3),range(3))
""")
        if st.button("Ejecutar y simular", key="t4_run", type="primary"):
            ns = {}
            try:
                exec(code, {"np":np, "QuantumCircuit":QuantumCircuit}, ns)
                qc = ns.get("qc", None)
                if qc is None:
                    st.warning("Tu código no definió `qc`."); st.stop()

                # Diagrama
                try:
                    figC = circuit_drawer(qc, output="mpl", style={'name':'mpl'})
                    st.pyplot(figC, use_container_width=True)
                    plt.close(figC)
                except Exception as e:
                    st.warning(f"No se pudo dibujar el circuito (mpl): {e}")

                # Simulación
                sim = Aer.get_backend("aer_simulator")
                qcsv = qc.remove_final_measurements(inplace=False)
                qcsv.save_statevector()
                sv = sim.run(transpile(qcsv, sim)).result().get_statevector()
                st.markdown("**Vector de estado (sin medidas)**")
                st.code(str(sv), language="text")

                if qc.num_clbits > 0:
                    qasm = Aer.get_backend("qasm_simulator")
                    counts = qasm.run(transpile(qc, qasm), shots=1024).result().get_counts()
                    try:
                        figH = plot_histogram(counts)
                        st.pyplot(figH, use_container_width=True)
                        plt.close(figH)
                    except Exception as e:
                        st.warning(f"No se pudo dibujar el histograma: {e}")

            except Exception as e:
                st.error(f"Error al ejecutar tu código: {e}")

st.success("App cargada correctamente.")
