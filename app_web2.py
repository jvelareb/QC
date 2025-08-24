# requirements: streamlit, numpy, matplotlib, qiskit, qiskit_aer, plotly
import streamlit as st
import numpy as np
import io
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.figure import Figure
import plotly.graph_objects as go
from matplotlib.lines import Line2D

# ----- Qiskit opcional (pestañas 3 y 4) -----
QISKIT_AVAILABLE = True
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import Aer
    from qiskit.visualization import circuit_drawer, plot_histogram, plot_state_qsphere
except Exception:
    QISKIT_AVAILABLE = False

# ⚠️ NO LLAMAR AQUÍ A set_page_config (ya se hace en secure_app.py)
# st.set_page_config(page_title="Quantum Toolkit (Streamlit)", layout="wide", page_icon="⚛️")

st.title("⚛️ Quantum Toolkit — Esfera de Bloch, Puertas y Circuitos (Streamlit)")

# ================= Utilidades 1 qubit =================
def ket_from_angles(theta_deg, phi_deg):
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    a = np.cos(th/2.0)
    b = np.exp(1j*ph)*np.sin(th/2.0)
    v = np.array([[a],[b]], dtype=complex)
    return v

def ket_from_amplitudes(alpha, beta):
    v = np.array([[alpha],[beta]], dtype=complex)
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

def fig_to_png_bytes(fig: Figure, dpi=800, *, bbox_inches="tight", pad_inches=0.02):
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )
    buf.seek(0)
    return buf

# ================= Controles combinados (slider + número) =================
def angle_input(label, key_base, min_val, max_val, *, step=0.1, unit="deg", default=0.0, disabled=False, help=None, fmt="%.2f"):
    """
    Slider + number_input sincronizados vía session_state.
    ✅ Evita warnings: no usamos 'value=' cuando hay 'key'.
    """
    slider_key = f"{key_base}_slider"
    input_key  = f"{key_base}_input"

    st.session_state.setdefault(slider_key, float(default))
    st.session_state.setdefault(input_key,  float(default))

    def _sync_from_slider():
        st.session_state[input_key] = float(st.session_state[slider_key])

    def _sync_from_input():
        v = float(st.session_state[input_key])
        v = max(float(min_val), min(float(max_val), v))
        st.session_state[input_key]  = v
        st.session_state[slider_key] = v

    csl, cnum = st.columns([4, 1])

    csl.slider(
        f"{label} ({unit})",
        min_val, max_val,
        step=step,
        key=slider_key,
        on_change=_sync_from_slider,
        disabled=disabled,
        help=help
    )

    cnum.number_input(
        " ",
        min_value=float(min_val),
        max_value=float(max_val),
        step=float(step),
        key=input_key,
        on_change=_sync_from_input,
        disabled=disabled,
        format=fmt
    )

    return float(st.session_state[slider_key])

# ============ Bloch: Plotly interactivo + Export Matplotlib ============
DEFAULT_CAMERA = dict(eye=dict(x=1.6, y=1.6, z=1.1))

def bloch_plotly(states, colors, labels, title, camera_key, height=640):
    cam = st.session_state.get(camera_key, DEFAULT_CAMERA)
    fig = go.Figure()

    # Superficie
    u, v = np.mgrid[0:2*np.pi:120j, 0:np.pi:120j]
    xs, ys, zs = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        showscale=False,
        opacity=0.12,
        colorscale=[[0, '#8ab4f8'], [1, '#8ab4f8']],
        hoverinfo='skip'
    ))

    # Ejes
    L = 1.1
    axes = {
        'x': ([-L, L], [0, 0], [0, 0]),
        'y': ([0, 0], [-L, L], [0, 0]),
        'z': ([0, 0], [0, 0], [-L, L])
    }
    for x,y,z in axes.values():
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines",
                                   line=dict(color="black", width=4),
                                   hoverinfo='skip', showlegend=False))

    # Marcas
    for t in [-1.0, -0.5, 0.5, 1.0]:
        for x,y,z in ([t,0,0],[0,t,0],[0,0,t]):
            fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='text',
                                       text=[f'{[x,y,z][0 if x else (1 if y else 2)]:.1f}'],
                                       showlegend=False, textfont=dict(color="gray", size=12),
                                       hoverinfo='skip'))

    # Etiquetas base
    lbl = 1.2
    labels_text = [
        (0,0, lbl,  r"$|0\rangle$"),
        (0,0,-lbl,  r"$|1\rangle$"),
        ( lbl,0,0,  r"$|+\rangle$"),
        (-lbl,0,0,  r"$|-\rangle$"),
        (0,  lbl,0, r"$|+i\rangle$"),
        (0, -lbl,0, r"$|-i\rangle$")
    ]
    for x,y,z,t in labels_text:
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='text',
                                   text=[t], textfont=dict(color="navy", size=14),
                                   showlegend=False, hoverinfo='skip'))

    # Vectores de estado
    for (psi, col, lab) in zip(states, colors, labels):
        x,y,z = bloch_xyz(psi)
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode="lines",
            line=dict(color=col, width=8),
            name=lab,
            hovertemplate=f"{lab}<br>x={x:.3f}<br>y={y:.3f}<br>z={z:.3f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="markers",
            marker=dict(color=col, size=6),
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        height=height,
        scene=dict(
            aspectmode='cube',
            camera=cam,
            xaxis=dict(range=[-1.1,1.1], tickvals=[-1,-0.5,0,0.5,1],
                       backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(range=[-1.1,1.1], tickvals=[-1,-0.5,0,0.5,1],
                       backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(range=[-1.1,1.1], tickvals=[-1,-0.5,0,0.5,1],
                       backgroundcolor="rgba(0,0,0,0)")
        ),
        margin=dict(l=4, r=4, b=4, t=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def draw_bloch_export(
    states, colors, labels, *,
    title="Esfera de Bloch", subtitle=None,
    figsize=(14.0, 9.4),
    sphere_ratio=9.0,
    proj_ratio=2.2,
    wspace=0.05,
    hspace=0.04
):
    fig = Figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[sphere_ratio, proj_ratio],
        hspace=hspace,
        wspace=wspace,
    )

    ax3d = fig.add_subplot(gs[0, :], projection="3d")
    ax_xy = fig.add_subplot(gs[1, 0], aspect="equal")
    ax_yz = fig.add_subplot(gs[1, 1], aspect="equal")
    ax_xz = fig.add_subplot(gs[1, 2], aspect="equal")

    # ===== Esfera 3D =====
    u, v = np.mgrid[0:2*np.pi:140j, 0:np.pi:140j]
    xs, ys, zs = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
    ax3d.plot_surface(xs, ys, zs, color="#8ab4f8", alpha=0.12, edgecolor="#789", linewidth=0.25)

    # Ejes
    L = 1.1
    ax3d.plot([-L, L], [0, 0], [0, 0], color="k", lw=1.0)
    ax3d.plot([0, 0], [-L, L], [0, 0], color="k", lw=1.0)
    ax3d.plot([0, 0], [0, 0], [-L, L], color="k", lw=1.0)

    # Marcas
    for t in (-1.0, -0.5, 0.5, 1.0):
        ax3d.text(t, 0, 0, f"{t:.1f}", color="gray", fontsize=8, ha="center", va="center")
        ax3d.text(0, t, 0, f"{t:.1f}", color="gray", fontsize=8, ha="center", va="center")
        ax3d.text(0, 0, t, f"{t:.1f}", color="gray", fontsize=8, ha="center", va="center")

    # Etiquetas base
    lbl = 1.2
    ax3d.text(0, 0,  lbl, r"$|0\rangle$",  color="navy", ha="center", va="center", fontsize=10)
    ax3d.text(0, 0, -lbl, r"$|1\rangle$",  color="navy", ha="center", va="center", fontsize=10)
    ax3d.text( lbl, 0, 0, r"$|+\rangle$",  color="navy", ha="center", va="center", fontsize=10)
    ax3d.text(-lbl, 0, 0, r"$|-\rangle$",  color="navy", ha="center", va="center", fontsize=10)
    ax3d.text(0,  lbl, 0, r"$|+i\rangle$", color="navy", ha="center", va="center", fontsize=10)
    ax3d.text(0, -lbl, 0, r"$|-i\rangle$", color="navy", ha="center", va="center", fontsize=10)

    # Vectores
    for (psi, col) in zip(states, colors):
        x, y, z = bloch_xyz(psi)
        ax3d.quiver(0, 0, 0, x, y, z, color=col, arrow_length_ratio=0.08, linewidth=2.4, length=1.0)
        ax3d.scatter([x],[y],[z], color=col, s=22, zorder=3)

    # Leyenda IN/OUT
    alias = {"|ψ_in⟩": "IN", "|ψ_out⟩": "OUT"}
    legend_labels = [alias.get(lab, lab) for lab in labels]
    handles = [Line2D([0],[0], color=col, lw=3) for col in colors]
    ax3d.legend(
        handles=handles, labels=legend_labels,
        loc="upper left", bbox_to_anchor=(0.02, 0.98),
        frameon=True, facecolor="white", edgecolor="#aaa",
        framealpha=0.85, fontsize=10, borderpad=0.25, handlelength=1.6
    )

    ax3d.set_title(title, pad=2)
    if subtitle:
        ax3d.text2D(0.5, 1.03, subtitle, transform=ax3d.transAxes,
                    ha="center", va="bottom", fontsize=11, color="black")
    ax3d.set_box_aspect((1,1,1))
    ax3d.set_axis_off()
    ax3d.view_init(elev=22, azim=30)

    # ===== Proyecciones =====
    def proj(ax, c1, c2, name, draw_theta=False, draw_phi=False, psi=None, col=None):
        circ = plt.Circle((0,0), 1.0, fill=False, ls="--", color="#789", linewidth=0.8)
        ax.add_patch(circ)
        ax.axhline(0, color="#aaa", lw=0.6)
        ax.axvline(0, color="#aaa", lw=0.6)
        ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02)
        ax.set_xlabel(c1, labelpad=1); ax.set_ylabel(c2, labelpad=1)
        ax.set_title(name, pad=2, fontsize=10)
        ax.margins(0); ax.grid(True, ls=":", alpha=0.35)

        for ps, cc in zip(states, colors):
            x, y, z = bloch_xyz(ps)
            data = {"x": x, "y": y, "z": z}
            ax.plot([0, data[c1]],[0, data[c2]], color=cc, lw=1.8)
            ax.scatter([data[c1]],[data[c2]], color=cc, s=26, zorder=3)

        if psi is not None:
            x, y, z = bloch_xyz(psi)
            th = np.arccos(np.clip(z, -1, 1))
            ph = np.arctan2(y, x)
            if draw_phi:
                ax.add_patch(Arc((0,0), 0.56, 0.56, angle=0,   theta1=0, theta2=np.rad2deg(ph), color=col, ls="--", lw=0.9))
                ax.text(0.37*np.cos(ph/2), 0.37*np.sin(ph/2), r"$\varphi$", color=col, fontsize=10)
            if draw_theta:
                ax.add_patch(Arc((0,0), 0.56, 0.56, angle=-90, theta1=0, theta2=np.rad2deg(th), color=col, ls="--", lw=0.9))
                ax.text(0.37*np.sin(th/2), 0.37*np.cos(th/2), r"$\theta$", color=col, fontsize=10)

    psi0 = states[0] if states else None
    col0 = colors[0] if colors else "C0"
    proj(ax_xy, "y","x", "Plano XY", draw_phi=True,  psi=psi0, col=col0)
    proj(ax_yz, "z","y", "Plano YZ", draw_theta=True, psi=psi0, col=col0)
    proj(ax_xz, "z","x", "Plano XZ")

    fig.subplots_adjust(left=0.02, right=0.995, top=0.985, bottom=0.06,
                        wspace=wspace, hspace=hspace)
    return fig

# ================== Circuitos predeterminados ==================
def qc_bell(measure=True):
    qc = QuantumCircuit(2,2 if measure else 0); qc.h(0); qc.cx(0,1)
    if measure: qc.measure([0,1],[0,1]); return qc
    return qc

def qc_ghz(n=3, measure=True):
    qc = QuantumCircuit(n, n if measure else 0); qc.h(0)
    for i in range(n-1): qc.cx(i,i+1)
    if measure: qc.measure(range(n), range(n)); return qc
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
        raise RuntimeError("Tu versión de Qiskit no soporta 'mcx'. Actualiza qiskit a 0.45+ o usa menos controles.")

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
        qc.append(
            c_amod15(a, 2**q),
            [q] + list(range(n_count, n_count+4))
        )
    # QFT^-1
    for j in range(n_count//2): qc.swap(j, n_count-1-j)
    for j in range(n_count):
        for m in range(j):
            qc.cp(-np.pi/2**(j-m), m, j)
        qc.h(j)
    qc.measure(range(n_count), range(n_count))
    return qc

# ========================= PESTAÑAS =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Estado 1-Qubit",
    "2. Puertas 1-Qubit",
    "3. Circuitos (Dashboard)",
    "4. Código Qiskit"
])

# ---------------- Pestaña 1 ----------------
with tab1:
    st.subheader("Esfera de Bloch interactiva y proyecciones")
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
            alpha, beta = mp[preset]
            psi1 = ket_from_amplitudes(alpha, beta)

        x,y,z = bloch_xyz(psi1)
        th, ph = angles_from_xyz(x,y,z)
        p0, p1 = abs(psi1[0,0])**2*100, abs(psi1[1,0])**2*100

        if st.button("🔄 Reset vista 3D", key="t1_reset"):
            st.session_state["t1_cam"] = DEFAULT_CAMERA
        fig_int = bloch_plotly([psi1], ['#d62728'], ['|ψ⟩'],
                               title="Esfera de Bloch (interactiva)",
                               camera_key="t1_cam", height=760)
        st.plotly_chart(fig_int, use_container_width=True, key="t1_plotly")

        st.caption(f"Ángulos: θ={th:.2f}°, φ={ph:.2f}°  |  Coords: x={x:.3f}, y={y:.3f}, z={z:.3f}  |  P(0)={p0:.1f}%, P(1)={p1:.1f}%")

    with right:
        fig_exp = draw_bloch_export(
            [psi1], ['#d62728'], ['|ψ⟩'],
            title="Esfera de Bloch y proyecciones",
            figsize=(15.0, 10.0),
            sphere_ratio=10.0,
            proj_ratio=2.0,
            hspace=0.03, wspace=0.04
        )
        st.pyplot(fig_exp, use_container_width=True)
        plt.close(fig_exp)  # 🔒 evita fugas de memoria
        st.download_button("⬇️ Descargar (PNG, 800 dpi)",
                           data=fig_to_png_bytes(fig_exp, 800),
                           file_name="bloch_estado.png",
                           mime="image/png",
                           key="t1_dl")

# ---------------- Pestaña 2 ----------------
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
            alpha, beta = mp[preset2]
            psi_in = ket_from_amplitudes(alpha, beta)

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
                                ['|ψ_in⟩','|ψ_out⟩'],
                                title=f"Esfera de Bloch — {gate_desc}",
                                camera_key="t2_cam", height=780)
        st.plotly_chart(fig2_int, use_container_width=True, key="t2_plotly")

    with right:
        fig2_exp = draw_bloch_export(
            [psi_in, psi_out], ['#1f77b4','#d62728'], ['|ψ_in⟩','|ψ_out⟩'],
            title=f"Esfera de Bloch — {gate_desc}",
            figsize=(14.0, 9.6),
            sphere_ratio=9.0,
            proj_ratio=2.0,
            hspace=0.04, wspace=0.05
        )
        st.pyplot(fig2_exp, use_container_width=True)
        plt.close(fig2_exp)
        st.download_button("⬇️ Descargar (PNG, 800 dpi)",
                           data=fig_to_png_bytes(fig2_exp, 800,pad_inches=0.02),
                           file_name="bloch_in_out.png",
                           mime="image/png",
                           key="t2_dl")

    xin,yin,zin = bloch_xyz(psi_in)
    xout,yout,zout = bloch_xyz(psi_out)
    th_in, ph_in   = angles_from_xyz(xin,yin,zin)
    th_out, ph_out = angles_from_xyz(xout,yout,zout)
    p0, p1 = abs(psi_out[0,0])**2*100, abs(psi_out[1,0])**2*100

    st.markdown("#### Resultado matemático")
    st.code(
        f"U = [[{pretty_c(U[0,0])}, {pretty_c(U[0,1])}],\n"
        f"     [{pretty_c(U[1,0])}, {pretty_c(U[1,1])}]]\n\n"
        f"|ψ_in⟩ = [{pretty_c(psi_in[0,0])}, {pretty_c(psi_in[1,0])}]^T\n"
        f"|ψ_out⟩ = U · |ψ_in⟩ = [{pretty_c(psi_out[0,0])}, {pretty_c(psi_out[1,0])}]^T",
        language="text"
    )
    st.caption(f"Ángulos entrada (θ={th_in:.2f}°, φ={ph_in:.2f}°) → salida (θ={th_out:.2f}°, φ={ph_out:.2f}°).  Probabilidades: P(0)={p0:.2f}%, P(1)={p1:.2f}%.")

# ---------------- Pestaña 3: Dashboard ----------------
with tab3:
    st.subheader("Circuitos predeterminados — Dashboard")
    if not QISKIT_AVAILABLE:
        st.error("Qiskit / qiskit_aer no están instalados.")
    else:
        left, right = st.columns([1,2], gap="large")

        with left:
            algo = st.selectbox("Algoritmo", [
                "Bell","GHZ","QFT","Deutsch-Jozsa","Bernstein-Vazirani","Grover","Shor (demo N=15)"
            ], key="t3_algo")

            with st.form("t3_param_form", clear_on_submit=False):
                if algo == "Bell":
                    measure = st.checkbox("Añadir medición", True, key="t3_bell_m")
                elif algo == "GHZ":
                    n = st.number_input("n (qubits ≥2)", min_value=2, value=3, step=1, key="t3_ghz_n")
                    measure = st.checkbox("Añadir medición", True, key="t3_ghz_m")
                elif algo == "QFT":
                    n = st.number_input("n (qubits ≥1)", min_value=1, value=3, step=1, key="t3_qft_n")
                    swaps = st.checkbox("SWAPs finales", True, key="t3_qft_s")
                    measure = st.checkbox("Añadir medición", True, key="t3_qft_m")
                elif algo == "Deutsch-Jozsa":
                    n = st.number_input("n (entrada ≥1)", min_value=1, value=3, step=1, key="t3_dj_n")
                    kind = st.selectbox("Oráculo", ["balanced","constant1"], key="t3_dj_kind")
                elif algo == "Bernstein-Vazirani":
                    s = st.text_input("Secreto s (binario)", "1011", key="t3_bv_s")
                elif algo == "Grover":
                    n = st.number_input("n (qubits ≥2)", min_value=2, value=3, step=1, key="t3_grover_n")
                    marked = st.text_input("Marcado (binario longitud n)", "111", key="t3_grover_marked")
                elif algo == "Shor (demo N=15)":
                    a = st.selectbox("a (coprimo con 15)", [2,7,8,11,13], key="t3_shor_a")

                shots = st.slider("Shots", 100, 8192, 1024, step=100, key="t3_shots")
                submitted = st.form_submit_button("Generar y simular", use_container_width=True)

        if "t3_qc" not in st.session_state:
            st.session_state.t3_qc = None
        if "t3_counts" not in st.session_state:
            st.session_state.t3_counts = None
        if "t3_sv" not in st.session_state:
            st.session_state.t3_sv = None

        if submitted:
            try:
                if algo == "Bell":
                    qc = qc_bell(measure)
                elif algo == "GHZ":
                    qc = qc_ghz(int(n), measure)
                elif algo == "QFT":
                    qc = qc_qft(int(n), swaps, measure)
                elif algo == "Deutsch-Jozsa":
                    qc = qc_deutsch_jozsa(int(n), kind)
                elif algo == "Bernstein-Vazirani":
                    if not all(c in "01" for c in s):
                        st.warning("El secreto s debe ser binario."); st.stop()
                    qc = qc_bernstein_vazirani(s)
                elif algo == "Grover":
                    if len(marked) != int(n) or not all(c in "01" for c in marked):
                        st.warning("Marcado debe ser binario de longitud n."); st.stop()
                    qc = qc_grover(int(n), marked)
                elif algo == "Shor (demo N=15)":
                    qc = qc_shor_demo_15(int(a))

                st.session_state.t3_qc = qc

                # Simulación (statevector)
                sim = Aer.get_backend("aer_simulator")
                qcsv = qc.remove_final_measurements(inplace=False)
                qcsv.save_statevector()
                sv = sim.run(transpile(qcsv, sim)).result().get_statevector()
                st.session_state.t3_sv = sv

                # Medidas
                counts = None
                if qc.num_clbits > 0:
                    qasm = Aer.get_backend("qasm_simulator")
                    counts = qasm.run(transpile(qc, qasm), shots=shots).result().get_counts()
                st.session_state.t3_counts = counts

            except Exception as e:
                st.error(f"Error al generar/simular: {e}")

        with right:
            dtabs = st.tabs(["Diagrama", "Medidas", "Q-sphere", "Statevector"])

            # Diagrama
            with dtabs[0]:
                if st.session_state.t3_qc is not None:
                    try:
                        figC = circuit_drawer(st.session_state.t3_qc, output="mpl", style={'name':'mpl'})
                        figC.set_size_inches(9, 3.2)
                        st.pyplot(figC, use_container_width=True)
                        plt.close(figC)
                        st.download_button("⬇️ Diagrama (PNG 800 dpi)",
                                           data=fig_to_png_bytes(figC, 800,pad_inches=0.02),
                                           file_name="circuito.png",
                                           mime="image/png",
                                           key="t3_dl_circ")
                    except Exception as e:
                        st.error(f"No se pudo dibujar el circuito: {e}")
                else:
                    st.info("Genera el circuito a la izquierda.")

            # Medidas
            with dtabs[1]:
                if st.session_state.t3_counts is not None:
                    figH = plot_histogram(st.session_state.t3_counts)
                    figH.set_size_inches(8.0, 3.0)
                    st.pyplot(figH, use_container_width=True)
                    plt.close(figH)
                    st.download_button("⬇️ Histograma (PNG 800 dpi)",
                                       data=fig_to_png_bytes(figH, 800,pad_inches=0.02),
                                       file_name="histograma.png",
                                       mime="image/png",
                                       key="t3_dl_hist")
                else:
                    st.info("Este circuito no tiene mediciones o no se ha simulado.")

            # Q-sphere
            with dtabs[2]:
                if st.session_state.t3_sv is not None:
                    figQ = plot_state_qsphere(Statevector(st.session_state.t3_sv))
                    figQ.set_size_inches(6.0, 6.0)
                    st.pyplot(figQ, use_container_width=False)
                    plt.close(figQ)
                    st.download_button("⬇️ Q-sphere (PNG 800 dpi)",
                                       data=fig_to_png_bytes(figQ, 800,pad_inches=0.02),
                                       file_name="qsphere.png",
                                       mime="image/png",
                                       key="t3_dl_q")
                else:
                    st.info("Vector de estado no disponible.")

            # Statevector (texto)
            with dtabs[3]:
                if st.session_state.t3_sv is not None:
                    st.code(str(st.session_state.t3_sv), language="text")
                else:
                    st.info("Vector de estado no disponible.")

# ---------------- Pestaña 4 ----------------
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
                figC = circuit_drawer(qc, output="mpl", style={'name':'mpl'})
                figC.set_size_inches(9, 3.2)
                st.pyplot(figC, use_container_width=True)
                plt.close(figC)

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
                    figH = plot_histogram(counts)
                    figH.set_size_inches(8.0, 3.0)
                    st.pyplot(figH, use_container_width=True)
                    plt.close(figH)
            except Exception as e:
                st.error(f"Error al ejecutar tu código: {e}")
