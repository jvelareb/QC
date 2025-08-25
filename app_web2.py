# -*- coding: utf-8 -*-
import io
import numpy as np
import streamlit as st

# Matplotlib en modo headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Arc
from matplotlib.lines import Line2D

from auth import login

# ================= Config básica =================
st.set_page_config(page_title="Quantum Toolkit", layout="wide", page_icon="⚛️")
st.title("⚛️ Quantum Toolkit — Esfera de Bloch, Puertas y Circuitos")
st.caption("UI estable: carga Qiskit bajo demanda para evitar bloqueos en despliegue.")

# ================= Login obligatorio =================
if not login():
    st.stop()

# ================= Utilidades comunes (sin Qiskit) =================
def ket_from_angles(theta_deg, phi_deg):
    th = np.deg2rad(theta_deg); ph = np.deg2rad(phi_deg)
    a = np.cos(th/2.0); b = np.exp(1j*ph)*np.sin(th/2.0)
    return np.array([[a],[b]], dtype=complex)

def ket_from_amplitudes(alpha, beta):
    v = np.array([[alpha],[beta]], complex); n = np.linalg.norm(v)
    return v if n < 1e-12 else v/n

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
    theta = np.rad2deg(np.arccos(np.clip(z,-1,1)))
    phi   = np.rad2deg(np.arctan2(y,x))
    return theta, phi

def U_gate(name, params=None):
    n = name.upper(); I = np.eye(2, complex)
    if n=="I": return I
    if n=="X": return np.array([[0,1],[1,0]], complex)
    if n=="Y": return np.array([[0,-1j],[1j,0]], complex)
    if n=="Z": return np.array([[1,0],[0,-1]], complex)
    if n=="H": return (1/np.sqrt(2))*np.array([[1,1],[1,-1]], complex)
    if n=="S": return np.array([[1,0],[0,1j]], complex)
    if n in ("S†","SDG"): return np.array([[1,0],[0,-1j]], complex)
    if n=="T": return np.array([[1,0],[0,np.exp(1j*np.pi/4)]], complex)
    if n in ("T†","TDG"): return np.array([[1,0],[0,np.exp(-1j*np.pi/4)]], complex)
    if n=="RX":
        theta=float(params or 0.0); c,s=np.cos(theta/2), -1j*np.sin(theta/2)
        return np.array([[c,s],[s,c]], complex)
    if n=="RY":
        theta=float(params or 0.0); c,s=np.cos(theta/2), np.sin(theta/2)
        return np.array([[c,-s],[s,c]], complex)
    if n=="RZ":
        theta=float(params or 0.0)
        return np.array([[np.exp(-1j*theta/2),0],[0,np.exp(1j*theta/2)]], complex)
    if n in ("U","U3"):
        theta,phi,lam = params or (0.0,0.0,0.0)
        return np.array([
            [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
            [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]
        ], complex)
    return I

def pretty_c(z, d=3):
    return f"{z.real:.{d}f}{z.imag:+.{d}f}j"

@st.cache_data(show_spinner=False)
def fig_to_png(_fig: Figure, dpi=800, *, bbox_inches="tight", pad_inches=0.02):
    """Exporta figura a PNG (cacheado). Ojo: argumento se llama _fig para evitar hashing de objetos no-hasheables."""
    buf = io.BytesIO()
    _fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    buf.seek(0)
    return buf

def angle_input(label, key_base, min_val, max_val, *, step=0.1, unit="deg", default=0.0):
    sk = f"{key_base}_slider"; nk=f"{key_base}_input"
    st.session_state.setdefault(sk, float(default))
    st.session_state.setdefault(nk, float(default))
    def _sync_s(): st.session_state[nk]=float(st.session_state[sk])
    def _sync_n():
        v=float(st.session_state[nk]); v=max(float(min_val),min(float(max_val),v))
        st.session_state[nk]=v; st.session_state[sk]=v
    c1,c2=st.columns([4,1])
    c1.slider(f"{label} ({unit})", min_val, max_val, step=step, key=sk, on_change=_sync_s)
    c2.number_input(" ", min_value=float(min_val), max_value=float(max_val), step=float(step), key=nk, on_change=_sync_n)
    return float(st.session_state[sk])

def draw_bloch_export(states, colors, labels, title="Esfera de Bloch", figsize=(14.0, 9.2)):
    fig = Figure(figsize=figsize)
    gs = fig.add_gridspec(2,3, height_ratios=[9,2.2], hspace=0.04, wspace=0.05)
    ax3d = fig.add_subplot(gs[0,:], projection="3d")
    ax_xy = fig.add_subplot(gs[1,0], aspect="equal")
    ax_yz = fig.add_subplot(gs[1,1], aspect="equal")
    ax_xz = fig.add_subplot(gs[1,2], aspect="equal")

    # esfera
    u,v = np.mgrid[0:2*np.pi:140j, 0:np.pi:140j]
    xs,ys,zs = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
    ax3d.plot_surface(xs,ys,zs, color="#8ab4f8", alpha=0.12, edgecolor="#789", linewidth=0.25)

    # ejes
    L=1.1
    ax3d.plot([-L,L],[0,0],[0,0], color="k", lw=1)
    ax3d.plot([0,0],[-L,L],[0,0], color="k", lw=1)
    ax3d.plot([0,0],[0,0],[-L,L], color="k", lw=1)

    # marcas
    for t in (-1.0,-0.5,0.5,1.0):
        ax3d.text(t,0,0,f"{t:.1f}", color="gray", fontsize=8, ha="center", va="center")
        ax3d.text(0,t,0,f"{t:.1f}", color="gray", fontsize=8, ha="center", va="center")
        ax3d.text(0,0,t,f"{t:.1f}", color="gray", fontsize=8, ha="center", va="center")

    # labels base
    lbl=1.2
    ax3d.text(0,0, lbl, r"$|0\rangle$", color="navy", fontsize=10, ha="center")
    ax3d.text(0,0,-lbl, r"$|1\rangle$", color="navy", fontsize=10, ha="center")
    ax3d.text( lbl,0,0, r"$|+\rangle$", color="navy", fontsize=10, ha="center")
    ax3d.text(-lbl,0,0, r"$|-\rangle$", color="navy", fontsize=10, ha="center")
    ax3d.text(0, lbl,0, r"$|+i\rangle$", color="navy", fontsize=10, ha="center")
    ax3d.text(0,-lbl,0, r"$|-i\rangle$", color="navy", fontsize=10, ha="center")

    # vectores
    for (psi, col) in zip(states, colors):
        x,y,z = bloch_xyz(psi)
        ax3d.quiver(0,0,0, x,y,z, color=col, arrow_length_ratio=0.08, linewidth=2.4, length=1.0)
        ax3d.scatter([x],[y],[z], color=col, s=22)

    handles=[Line2D([0],[0], color=c, lw=3) for c in colors]
    ax3d.legend(handles, labels, loc="upper left")
    ax3d.set_title(title, pad=2); ax3d.set_box_aspect((1,1,1)); ax3d.set_axis_off(); ax3d.view_init(elev=22, azim=30)

    # proyecciones
    def proj(ax, c1, c2, name, psi=None, col=None, draw_theta=False, draw_phi=False):
        circ = plt.Circle((0,0), 1.0, fill=False, ls="--", color="#789", linewidth=0.8); ax.add_patch(circ)
        ax.axhline(0, color="#aaa", lw=0.6); ax.axvline(0, color="#aaa", lw=0.6)
        ax.set_xlim(-1.02,1.02); ax.set_ylim(-1.02,1.02)
        ax.set_title(name, pad=2, fontsize=10); ax.grid(True, ls=":", alpha=0.35)
        for ps, cc in zip(states, colors):
            x,y,z = bloch_xyz(ps); data={"x":x,"y":y,"z":z}
            ax.plot([0, data[c1]],[0, data[c2]], color=cc, lw=1.8)
            ax.scatter([data[c1]],[data[c2]], color=cc, s=26)
        if psi is not None:
            x,y,z = bloch_xyz(psi)
            th=np.arccos(np.clip(z,-1,1)); ph=np.arctan2(y,x)
            if draw_phi: ax.add_patch(Arc((0,0),0.56,0.56, angle=0,   theta1=0, theta2=np.rad2deg(ph), color=col, ls="--", lw=0.9))
            if draw_theta: ax.add_patch(Arc((0,0),0.56,0.56, angle=-90, theta1=0, theta2=np.rad2deg(th), color=col, ls="--", lw=0.9))

    psi0 = states[0] if states else None
    col0 = colors[0] if colors else "C0"
    proj(ax_xy, "y","x", "Plano XY", psi=psi0, col=col0, draw_phi=True)
    proj(ax_yz, "z","y", "Plano YZ", psi=psi0, col=col0, draw_theta=True)
    proj(ax_xz, "z","x", "Plano XZ")

    fig.subplots_adjust(left=0.02, right=0.995, top=0.985, bottom=0.06, wspace=0.05, hspace=0.04)
    return fig

# ===================== TABS =====================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Estado 1-Qubit",
    "2. Puertas 1-Qubit",
    "3. Circuitos (Qiskit)",
    "4. Código Qiskit"
])

# --------- Tab 1 ----------
with tab1:
    st.subheader("Esfera de Bloch — estado único")
    method = st.radio("Método", ["Ángulos (θ, φ)","Amplitudes (α, β)","Presets"], horizontal=True, key="t1_method")
    if method=="Ángulos (θ, φ)":
        th = angle_input("θ","t1_theta",0,180,default=0.0)
        ph = angle_input("φ","t1_phi",0,360,default=0.0)
        psi = ket_from_angles(th, ph)
    elif method=="Amplitudes (α, β)":
        a_re = st.number_input("α (real)", value=1.0, key="t1_are")
        a_im = st.number_input("α (imag)", value=0.0, key="t1_aim")
        b_re = st.number_input("β (real)", value=0.0, key="t1_bre")
        b_im = st.number_input("β (imag)", value=0.0, key="t1_bim")
        psi = ket_from_amplitudes(a_re+1j*a_im, b_re+1j*b_im)
    else:
        preset = st.selectbox("Preset", ["|0⟩","|1⟩","|+⟩","|−⟩","|+i⟩","|−i⟩"], key="t1_preset")
        mp={"|0⟩":(1,0),"|1⟩":(0,1),"|+⟩":(1/np.sqrt(2),1/np.sqrt(2)),
            "|−⟩":(1/np.sqrt(2),-1/np.sqrt(2)),"|+i⟩":(1/np.sqrt(2),1j/np.sqrt(2)),
            "|−i⟩":(1/np.sqrt(2),-1j/np.sqrt(2))}
        alpha,beta = mp[preset]; psi = ket_from_amplitudes(alpha,beta)

    x,y,z = bloch_xyz(psi)
    th2,ph2 = angles_from_xyz(x,y,z)
    p0,p1 = abs(psi[0,0])**2*100, abs(psi[1,0])**2*100

    fig = draw_bloch_export([psi], ['#d62728'], ['|ψ⟩'], title="Esfera de Bloch y proyecciones")
    st.pyplot(fig, use_container_width=True)
    st.download_button("⬇️ PNG (Bloch)", data=fig_to_png(fig), file_name="bloch.png", mime="image/png")

    st.caption(f"Ángulos: θ={th2:.2f}°, φ={ph2:.2f}° | x={x:.3f}, y={y:.3f}, z={z:.3f} | P(0)={p0:.1f}%, P(1)={p1:.1f}%")

# --------- Tab 2 ----------
with tab2:
    st.subheader("Puertas 1-Qubit: |ψ_in⟩ → |ψ_out⟩")

    m2 = st.radio("Método", ["Ángulos (θ, φ)","Amplitudes (α, β)","Presets"], horizontal=True, key="t2_method")
    if m2=="Ángulos (θ, φ)":
        th = angle_input("θ_in","t2_theta",0,180,default=0.0)
        ph = angle_input("φ_in","t2_phi",0,360,default=0.0)
        psi_in = ket_from_angles(th, ph)
    elif m2=="Amplitudes (α, β)":
        a_re = st.number_input("α (real)", value=1.0, key="t2_are")
        a_im = st.number_input("α (imag)", value=0.0, key="t2_aim")
        b_re = st.number_input("β (real)", value=0.0, key="t2_bre")
        b_im = st.number_input("β (imag)", value=0.0, key="t2_bim")
        psi_in = ket_from_amplitudes(a_re+1j*a_im, b_re+1j*b_im)
    else:
        preset = st.selectbox("Preset", ["|0⟩","|1⟩","|+⟩","|−⟩","|+i⟩","|−i⟩"], key="t2_preset")
        mp={"|0⟩":(1,0),"|1⟩":(0,1),"|+⟩":(1/np.sqrt(2),1/np.sqrt(2)),
            "|−⟩":(1/np.sqrt(2),-1/np.sqrt(2)),"|+i⟩":(1/np.sqrt(2),1j/np.sqrt(2)),
            "|−i⟩":(1/np.sqrt(2),-1j/np.sqrt(2))}
        alpha,beta = mp[preset]; psi_in = ket_from_amplitudes(alpha,beta)

    gname = st.selectbox("Puerta", ["I","X","Y","Z","H","S","S†","T","T†","Rx","Ry","Rz","U(θ,φ,λ)"], key="t2_gate")
    if gname=="Rx":
        theta = st.number_input("θ Rx (rad)", value=float(np.pi/2), step=0.01, format="%.3f")
        U = U_gate("RX", theta); desc=f"Rx({theta:.3f})"
    elif gname=="Ry":
        theta = st.number_input("θ Ry (rad)", value=float(np.pi/2), step=0.01, format="%.3f")
        U = U_gate("RY", theta); desc=f"Ry({theta:.3f})"
    elif gname=="Rz":
        theta = st.number_input("θ Rz (rad)", value=float(np.pi/2), step=0.01, format="%.3f")
        U = U_gate("RZ", theta); desc=f"Rz({theta:.3f})"
    elif gname=="U(θ,φ,λ)":
        td = angle_input("θ U","t2_U_theta",-360,360,default=90.0)
        pd = angle_input("φ U","t2_U_phi",-360,360,default=0.0)
        ld = angle_input("λ U","t2_U_lam",-360,360,default=0.0)
        U = U_gate("U", (np.deg2rad(td), np.deg2rad(pd), np.deg2rad(ld)))
        desc=f"U(θ={td:.1f}, φ={pd:.1f}, λ={ld:.1f})"
    else:
        U = U_gate(gname); desc=gname

    psi_out = U @ psi_in

    fig2 = draw_bloch_export([psi_in, psi_out], ['#1f77b4','#d62728'], ['|ψ_in⟩','|ψ_out⟩'], title=f"Esfera de Bloch — {desc}")
    st.pyplot(fig2, use_container_width=True)
    st.download_button("⬇️ PNG (Puerta)", data=fig_to_png(fig2), file_name="bloch_gate.png", mime="image/png")

    xin,yin,zin = bloch_xyz(psi_in)
    xout,yout,zout = bloch_xyz(psi_out)
    th_i,ph_i = angles_from_xyz(xin,yin,zin)
    th_o,ph_o = angles_from_xyz(xout,yout,zout)
    st.caption(f"Entrada θ={th_i:.2f}°, φ={ph_i:.2f}° → Salida θ={th_o:.2f}°, φ={ph_o:.2f}°")

# --------- Helpers para cargar Qiskit una sola vez ---------
def ensure_qiskit_loaded():
    if st.session_state.get("qiskit_loaded"):
        return True
    return False

def load_qiskit_now():
    if st.session_state.get("qiskit_loaded"):
        return
    # Import pesado bajo demanda
    from qiskit import QuantumCircuit, transpile  # noqa: F401
    from qiskit.quantum_info import Statevector    # noqa: F401
    from qiskit_aer import Aer                     # noqa: F401
    from qiskit.visualization import circuit_drawer, plot_histogram, plot_state_qsphere  # noqa: F401
    st.session_state["qiskit_loaded"] = True

# --------- Tab 3: Circuitos predeterminados (Qiskit) ----------
with tab3:
    st.subheader("Circuitos predeterminados (Qiskit) — carga bajo demanda")

    if not ensure_qiskit_loaded():
        if st.button("⚙️ Cargar motor Qiskit", type="primary"):
            with st.spinner("Inicializando Qiskit…"):
                try:
                    load_qiskit_now()
                except Exception as e:
                    st.error("No se pudo inicializar Qiskit.")
                    st.exception(e)
        st.info("Pulsa el botón para cargar Qiskit y evitar bloqueos al arrancar.")
    else:
        # Ya está cargado: ahora podemos importar y usar
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import Statevector
        from qiskit_aer import Aer
        from qiskit.visualization import circuit_drawer, plot_histogram, plot_state_qsphere

        # ==== Definición de circuitos ====
        def qc_bell(measure=True):
            qc = QuantumCircuit(2,2 if measure else 0); qc.h(0); qc.cx(0,1)
            if measure: qc.measure([0,1],[0,1])
            return qc

        def qc_ghz(n=3, measure=True):
            qc = QuantumCircuit(n, n if measure else 0); qc.h(0)
            for i in range(n-1): qc.cx(i,i+1)
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
                raise RuntimeError("Tu versión de Qiskit no soporta 'mcx'. Actualiza qiskit o reduce controles.")

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

        # ==== Controles UI ====
        left, right = st.columns([1,2], gap="large")
        with left:
            algo = st.selectbox("Algoritmo", [
                "Bell","GHZ","QFT","Deutsch-Jozsa","Bernstein-Vazirani","Grover","Shor (demo N=15)"
            ], key="t3_algo")
            with st.form("t3_form"):
                params = {}
                if algo=="GHZ":
                    params["n"] = st.number_input("n (≥2)", min_value=2, value=3, step=1)
                    params["measure"] = st.checkbox("Mediciones", True)
                elif algo=="QFT":
                    params["n"] = st.number_input("n (≥1)", min_value=1, value=3, step=1)
                    params["swaps"] = st.checkbox("SWAPs finales", True)
                    params["measure"] = st.checkbox("Mediciones", True)
                elif algo=="Deutsch-Jozsa":
                    params["n"] = st.number_input("n (entrada ≥1)", min_value=1, value=3, step=1)
                    params["kind"] = st.selectbox("Oráculo", ["balanced","constant1"])
                elif algo=="Bernstein-Vazirani":
                    params["s"] = st.text_input("Secreto s (binario)", "1011")
                elif algo=="Grover":
                    params["n"] = st.number_input("n (≥2)", min_value=2, value=3, step=1)
                    params["marked"] = st.text_input("Marcado (binario, long n)", "111")
                elif algo=="Shor (demo N=15)":
                    params["a"] = st.selectbox("a (coprimo con 15)", [2,7,8,11,13])
                else:
                    params["measure"] = st.checkbox("Mediciones", True)
                shots = st.slider("Shots", 100, 8192, 1024, step=100)
                submitted = st.form_submit_button("Generar y simular", use_container_width=True)

        if submitted:
            try:
                # Construcción QC según algoritmo
                if algo=="Bell":
                    qc = qc_bell(params.get("measure", True))
                elif algo=="GHZ":
                    qc = qc_ghz(int(params["n"]), params.get("measure", True))
                elif algo=="QFT":
                    qc = qc_qft(int(params["n"]), params.get("swaps", True), params.get("measure", True))
                elif algo=="Deutsch-Jozsa":
                    qc = qc_deutsch_jozsa(int(params["n"]), params.get("kind","balanced"))
                elif algo=="Bernstein-Vazirani":
                    s = params["s"]
                    if not all(c in "01" for c in s):
                        st.warning("El secreto s debe ser binario."); st.stop()
                    qc = qc_bernstein_vazirani(s)
                elif algo=="Grover":
                    n = int(params["n"]); marked = params["marked"]
                    if len(marked)!=n or not all(c in "01" for c in marked):
                        st.warning("Marcado debe ser binario de longitud n."); st.stop()
                    qc = qc_grover(n, marked)
                elif algo=="Shor (demo N=15)":
                    qc = qc_shor_demo_15(int(params["a"]))
                else:
                    qc = qc_bell(True)

                st.session_state["qc"] = qc

                # Simulación
                sim = Aer.get_backend("aer_simulator")
                qcsv = qc.remove_final_measurements(inplace=False)
                qcsv.save_statevector()
                sv = sim.run(transpile(qcsv, sim)).result().get_statevector()
                st.session_state["sv"] = sv

                counts = None
                if qc.num_clbits > 0:
                    qasm = Aer.get_backend("qasm_simulator")
                    counts = qasm.run(transpile(qc, qasm), shots=shots).result().get_counts()
                st.session_state["counts"] = counts

            except Exception as e:
                st.error("Error al generar/simular:")
                st.exception(e)

        with right:
            dt1, dt2, dt3, dt4 = st.tabs(["Diagrama", "Medidas", "Q-sphere", "Statevector"])

            with dt1:
                qc = st.session_state.get("qc")
                if qc is not None:
                    try:
                        figC = circuit_drawer(qc, output="mpl", style={'name':'mpl'})
                        st.pyplot(figC, use_container_width=True)
                        st.download_button("⬇️ Diagrama (PNG)", data=fig_to_png(figC), file_name="circuito.png", mime="image/png")
                        plt.close(figC)
                    except Exception as e:
                        st.error("No se pudo dibujar el circuito (instala 'pylatexenc').")
                        st.exception(e)
                else:
                    st.info("Genera el circuito en la izquierda.")

            with dt2:
                counts = st.session_state.get("counts")
                if counts is not None:
                    try:
                        from qiskit.visualization import plot_histogram
                        figH = plot_histogram(counts)
                        st.pyplot(figH, use_container_width=True)
                        st.download_button("⬇️ Histograma (PNG)", data=fig_to_png(figH), file_name="histograma.png", mime="image/png")
                        plt.close(figH)
                    except Exception as e:
                        st.error("No se pudo generar el histograma.")
                        st.exception(e)
                else:
                    st.info("Sin mediciones o sin simular.")

            with dt3:
                sv = st.session_state.get("sv")
                if sv is not None:
                    try:
                        from qiskit.visualization import plot_state_qsphere
                        figQ = plot_state_qsphere(Statevector(sv))
                        st.pyplot(figQ, use_container_width=True)
                        st.download_button("⬇️ Q-sphere (PNG)", data=fig_to_png(figQ), file_name="qsphere.png", mime="image/png")
                        plt.close(figQ)
                    except Exception as e:
                        st.error("No se pudo generar la Q-sphere.")
                        st.exception(e)
                else:
                    st.info("Vector de estado no disponible.")

            with dt4:
                sv = st.session_state.get("sv")
                if sv is not None:
                    st.code(str(sv), language="text")
                else:
                    st.info("Vector de estado no disponible.")

# --------- Tab 4: Editor Qiskit ----------
with tab4:
    st.subheader("Ejecutor de código Qiskit (define una variable `qc`)")
    if not ensure_qiskit_loaded():
        if st.button("⚙️ Cargar Qiskit para el editor"):
            with st.spinner("Inicializando Qiskit…"):
                try:
                    load_qiskit_now()
                except Exception as e:
                    st.error("No se pudo inicializar Qiskit.")
                    st.exception(e)
        st.info("Pulsa el botón para cargar Qiskit y usar el editor.")
    else:
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import Statevector
        from qiskit_aer import Aer
        from qiskit.visualization import circuit_drawer, plot_histogram

        code = st.text_area("Código Python", height=220, value=
"""from qiskit import QuantumCircuit
# Ejemplo: GHZ de 3 qubits
qc = QuantumCircuit(3,3)
qc.h(0); qc.cx(0,1); qc.cx(0,2); qc.measure_all()
""")
        if st.button("Ejecutar y simular", type="primary"):
            try:
                ns={}
                exec(code, {"np":np, "QuantumCircuit":QuantumCircuit}, ns)
                qc = ns.get("qc")
                if qc is None:
                    st.warning("Tu código no definió `qc`."); st.stop()

                figC = circuit_drawer(qc, output="mpl", style={'name':'mpl'})
                st.pyplot(figC, use_container_width=True)
                st.download_button("⬇️ Diagrama (PNG)", data=fig_to_png(figC), file_name="editor_circuito.png", mime="image/png")
                plt.close(figC)

                sim = Aer.get_backend("aer_simulator")
                qcsv = qc.remove_final_measurements(inplace=False); qcsv.save_statevector()
                sv = sim.run(transpile(qcsv, sim)).result().get_statevector()
                st.markdown("**Statevector (sin medidas):**")
                st.code(str(sv), language="text")

                if qc.num_clbits>0:
                    qasm = Aer.get_backend("qasm_simulator")
                    counts = qasm.run(transpile(qc, qasm), shots=1024).result().get_counts()
                    figH = plot_histogram(counts)
                    st.pyplot(figH, use_container_width=True)
                    st.download_button("⬇️ Histograma (PNG)", data=fig_to_png(figH), file_name="editor_hist.png", mime="image/png")
                    plt.close(figH)
            except Exception as e:
                st.error("Error al ejecutar tu código:")
                st.exception(e)
