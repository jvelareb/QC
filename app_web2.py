# app_web2.py — versión estable con Bloch (Matplotlib) + proyecciones y fallback de circuit_drawer
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Arc
import streamlit as st

# Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

# ===== pylatexenc es opcional (para circuit_drawer con 'mpl')
try:
    from qiskit.visualization import circuit_drawer
    _HAS_PYLATEXENC = True
except Exception:
    _HAS_PYLATEXENC = False


# ================= Utilidades básicas =================
@st.cache_resource(show_spinner=False)
def get_backends():
    return {
        "aer": Aer.get_backend("aer_simulator"),
        "qasm": Aer.get_backend("qasm_simulator"),
    }

@st.cache_data(show_spinner=False)
def fig_to_png_bytes(_fig: Figure, dpi=800, *, bbox_inches="tight", pad_inches=0.02):
    """Exporta Figure a PNG en memoria (el guion bajo evita errores de hash en cache)."""
    buf = io.BytesIO()
    _fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    buf.seek(0)
    return buf


# ================= Bloch helpers (Matplotlib puro) =================
def ket_from_angles(theta_deg: float, phi_deg: float):
    th = np.deg2rad(theta_deg); ph = np.deg2rad(phi_deg)
    a = np.cos(th/2.0); b = np.exp(1j*ph)*np.sin(th/2.0)
    v = np.array([a, b], dtype=complex)
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)

def bloch_xyz_from_statevector(sv: Statevector):
    a, b = sv.data[0], sv.data[1]
    x = 2*np.real(np.conj(a)*b)
    y = 2*np.imag(np.conj(b)*a)   # == -2*Im(conj(a)*b)
    z = np.abs(a)**2 - np.abs(b)**2
    return float(x), float(y), float(z)

def _draw_bloch_axes(ax):
    # Esfera
    u = np.linspace(0, 2*np.pi, 140)
    v = np.linspace(0, np.pi, 140)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="#8ab4f8", alpha=0.12, edgecolor="#789", linewidth=0.25, zorder=0)
    # Ejes
    L = 1.1
    ax.plot([-L,L],[0,0],[0,0], color="k", lw=1.0)
    ax.plot([0,0],[-L,L],[0,0], color="k", lw=1.0)
    ax.plot([0,0],[0,0],[-L,L], color="k", lw=1.0)
    # Marcas
    for t in (-1.0,-0.5,0.5,1.0):
        ax.text(t,0,0,f"{t:.1f}", color="gray", fontsize=8, ha="center", va="center")
        ax.text(0,t,0,f"{t:.1f}", color="gray", fontsize=8, ha="center", va="center")
        ax.text(0,0,t,f"{t:.1f}", color="gray", fontsize=8, ha="center", va="center")
    # Etiquetas base
    lbl = 1.2
    ax.text(0,0, lbl,  r"$|0\rangle$",  color="navy", ha="center", va="center", fontsize=10)
    ax.text(0,0,-lbl,  r"$|1\rangle$",  color="navy", ha="center", va="center", fontsize=10)
    ax.text( lbl,0,0,  r"$|+\rangle$",  color="navy", ha="center", va="center", fontsize=10)
    ax.text(-lbl,0,0,  r"$|-\rangle$",  color="navy", ha="center", va="center", fontsize=10)
    ax.text(0, lbl,0,  r"$|+i\rangle$", color="navy", ha="center", va="center", fontsize=10)
    ax.text(0,-lbl,0,  r"$|-i\rangle$", color="navy", ha="center", va="center", fontsize=10)
    # Estética
    ax.set_box_aspect((1,1,1))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=30)

def draw_bloch_matplotlib(sv: Statevector, *, figsize=(6,6), title="Esfera de Bloch", color="#d62728"):
    x, y, z = bloch_xyz_from_statevector(sv)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    _draw_bloch_axes(ax)
    # vector
    ax.quiver(0,0,0, x,y,z, color=color, arrow_length_ratio=0.08, linewidth=2.4, length=1.0, zorder=3)
    ax.scatter([x],[y],[z], color=color, s=28, zorder=4)
    ax.set_title(title, pad=6)
    fig.tight_layout()
    return fig

def draw_bloch_with_projections(sv: Statevector, *, figsize=(14.0, 9.4),
                                sphere_ratio=9.0, proj_ratio=2.2, title="Esfera de Bloch y proyecciones",
                                color="#d62728"):
    """Esfera 3D + proyecciones XY, YZ, XZ (como tu versión antigua)."""
    x, y, z = bloch_xyz_from_statevector(sv)
    from matplotlib.lines import Line2D

    fig = Figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[sphere_ratio, proj_ratio],
        hspace=0.04, wspace=0.05,
    )
    ax3d = fig.add_subplot(gs[0, :], projection="3d")
    _draw_bloch_axes(ax3d)
    ax3d.quiver(0,0,0, x,y,z, color=color, arrow_length_ratio=0.08, linewidth=2.4, length=1.0, zorder=3)
    ax3d.scatter([x],[y],[z], color=color, s=28, zorder=4)
    ax3d.set_title(title, pad=4)

    def _proj(ax, c1, c2, name, draw_theta=False, draw_phi=False):
        circ = plt.Circle((0,0), 1.0, fill=False, ls="--", color="#789", linewidth=0.8)
        ax.add_patch(circ)
        ax.axhline(0, color="#aaa", lw=0.6)
        ax.axvline(0, color="#aaa", lw=0.6)
        ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02)
        ax.set_xlabel(c1, labelpad=1); ax.set_ylabel(c2, labelpad=1)
        ax.set_title(name, pad=2, fontsize=10)
        ax.margins(0); ax.grid(True, ls=":", alpha=0.35)
        data = {"x": x, "y": y, "z": z}
        ax.plot([0, data[c1]],[0, data[c2]], color=color, lw=1.8)
        ax.scatter([data[c1]],[data[c2]], color=color, s=26, zorder=3)
        # ángulos (theta/phi) opcionales
        th = np.arccos(np.clip(z, -1, 1))
        ph = np.arctan2(y, x)
        if draw_phi:
            ax.add_patch(Arc((0,0), 0.56, 0.56, angle=0,   theta1=0, theta2=np.rad2deg(ph), color=color, ls="--", lw=0.9))
            ax.text(0.37*np.cos(ph/2), 0.37*np.sin(ph/2), r"$\varphi$", color=color, fontsize=10)
        if draw_theta:
            ax.add_patch(Arc((0,0), 0.56, 0.56, angle=-90, theta1=0, theta2=np.rad2deg(th), color=color, ls="--", lw=0.9))
            ax.text(0.37*np.sin(th/2), 0.37*np.cos(th/2), r"$\theta$", color=color, fontsize=10)

    ax_xy = fig.add_subplot(gs[1, 0], aspect="equal")
    ax_yz = fig.add_subplot(gs[1, 1], aspect="equal")
    ax_xz = fig.add_subplot(gs[1, 2], aspect="equal")
    _proj(ax_xy, "y","x", "Plano XY", draw_phi=True)
    _proj(ax_yz, "z","y", "Plano YZ", draw_theta=True)
    _proj(ax_xz, "z","x", "Plano XZ")

    fig.subplots_adjust(left=0.02, right=0.995, top=0.985, bottom=0.06)
    return fig


# ================= Puertas 1-qubit =================
def apply_single_qubit_gate(sv: Statevector, gate: str, param=None) -> Statevector:
    qc = QuantumCircuit(1)
    g = gate.upper()
    if g == "I": pass
    elif g == "X": qc.x(0)
    elif g == "Y": qc.y(0)
    elif g == "Z": qc.z(0)
    elif g == "H": qc.h(0)
    elif g == "S": qc.s(0)
    elif g in ("S†","SDG"): qc.sdg(0)
    elif g == "T": qc.t(0)
    elif g in ("T†","TDG"): qc.tdg(0)
    elif g == "RX": qc.rx(float(param or 0.0), 0)
    elif g == "RY": qc.ry(float(param or 0.0), 0)
    elif g == "RZ": qc.rz(float(param or 0.0), 0)
    elif g in ("U","U3"):
        theta, phi, lam = param if isinstance(param, tuple) else (0.0,0.0,0.0)
        qc.u(theta, phi, lam, 0)
    return sv.evolve(qc)


# ================= Interfaz principal =================
def run_app():
    st.title("⚛️ Quantum Toolkit — Matplotlib + Qiskit (estable)")

    tab1, tab2, tab3, tab4 = st.tabs([
        "1) Bloch 1-Qubit",
        "2) Puertas 1-Qubit",
        "3) Circuitos (Dashboard)",
        "4) Código Qiskit",
    ])

    # ---------- TAB 1: Bloch ----------
    with tab1:
        st.subheader("Esfera de Bloch con proyecciones (Matplotlib)")
        c1, c2 = st.columns(2)
        with c1:
            th = st.slider("θ (deg)", 0.0, 180.0, 45.0, 0.1, key="t1_theta")
        with c2:
            ph = st.slider("φ (deg)", 0.0, 360.0, 30.0, 0.1, key="t1_phi")

        sv = Statevector(ket_from_angles(th, ph))

        try:
            fig = draw_bloch_with_projections(sv, figsize=(15.0, 10.0), title="Esfera de Bloch y proyecciones", color="#d62728")
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇️ Descargar (PNG, 800 dpi)", data=fig_to_png_bytes(fig), file_name="bloch_estado.png", mime="image/png")
            plt.close(fig)
        except Exception as e:
            st.error("No se pudo renderizar la Bloch.")
            st.exception(e)

        # Probabilidades
        p0 = abs(sv.data[0])**2 * 100
        p1 = abs(sv.data[1])**2 * 100
        st.caption(f"P(0)={p0:.2f}%, P(1)={p1:.2f}%")

    # ---------- TAB 2: Puertas ----------
    with tab2:
        st.subheader("Puertas 1-Qubit: |ψ_in⟩ → |ψ_out⟩")
        left, right = st.columns([1,1], gap="large")

        with left:
            mode = st.radio("Estado inicial", ["Ángulos (θ, φ)", "Presets"], horizontal=True, key="t2_mode")
            if mode == "Ángulos (θ, φ)":
                th2 = st.slider("θ_in (deg)", 0.0, 180.0, 0.0, 0.1, key="t2_theta")
                ph2 = st.slider("φ_in (deg)", 0.0, 360.0, 0.0, 0.1, key="t2_phi")
                sv_in = Statevector(ket_from_angles(th2, ph2))
            else:
                preset = st.selectbox("Preset", ["|0⟩","|1⟩","|+⟩","|−⟩","|+i⟩","|−i⟩"], key="t2_preset")
                mp = {
                    "|0⟩": np.array([1,0],complex),
                    "|1⟩": np.array([0,1],complex),
                    "|+⟩": (1/np.sqrt(2))*np.array([1,1],complex),
                    "|−⟩": (1/np.sqrt(2))*np.array([1,-1],complex),
                    "|+i⟩": (1/np.sqrt(2))*np.array([1,1j],complex),
                    "|−i⟩": (1/np.sqrt(2))*np.array([1,-1j],complex),
                }
                sv_in = Statevector(mp[preset])

            gname = st.selectbox("Puerta", ["I","X","Y","Z","H","S","S†","T","T†","Rx","Ry","Rz","U(θ,φ,λ)"], key="t2_gate")
            param = None
            if gname in ("Rx","Ry","Rz"):
                param = st.number_input("θ (rad)", value=float(np.pi/2), step=0.01, format="%.3f", key="t2_param")
            elif gname == "U(θ,φ,λ)":
                t = st.number_input("θ (deg)", value=90.0, step=1.0, key="t2_u_t")
                p = st.number_input("φ (deg)", value=0.0, step=1.0, key="t2_u_p")
                l = st.number_input("λ (deg)", value=0.0, step=1.0, key="t2_u_l")
                param = (np.deg2rad(t), np.deg2rad(p), np.deg2rad(l))

        try:
            sv_out = apply_single_qubit_gate(sv_in, gname, param)
        except Exception as e:
            sv_out = sv_in
            st.error("Error aplicando la puerta.")
            st.exception(e)

        with right:
            cols = st.columns(2)
            try:
                fig_in  = draw_bloch_matplotlib(sv_in,  figsize=(5.4,5.4), title="IN",  color="#1f77b4")
                cols[0].pyplot(fig_in, use_container_width=True);  plt.close(fig_in)
            except Exception as e:
                cols[0].error("Error Bloch IN"); cols[0].exception(e)
            try:
                fig_out = draw_bloch_matplotlib(sv_out, figsize=(5.4,5.4), title="OUT", color="#d62728")
                cols[1].pyplot(fig_out, use_container_width=True); plt.close(fig_out)
            except Exception as e:
                cols[1].error("Error Bloch OUT"); cols[1].exception(e)

        p0 = abs(sv_out.data[0])**2 * 100
        p1 = abs(sv_out.data[1])**2 * 100
        st.markdown(f"**Probabilidades salida**: P(0)={p0:.2f}%, P(1)={p1:.2f}%")

    # ---------- TAB 3: Circuitos ----------
    with tab3:
        st.subheader("Circuitos predeterminados — Simulación (Aer)")
        b = get_backends()
        left, right = st.columns([1,2], gap="large")

        with left:
            algo = st.selectbox("Algoritmo", [
                "Bell","GHZ","QFT","Deutsch-Jozsa","Bernstein-Vazirani","Grover demo","Shor demo N=15"
            ], key="t3_algo")

            params = {}
            if algo in ("GHZ","QFT"):
                params["n"] = st.number_input("n (qubits)", min_value=2, value=3, step=1, key="t3_n")
            if algo == "Bernstein-Vazirani":
                params["s"] = st.text_input("Secreto s (binario)", "1011", key="t3_bv_s")
            if algo == "Deutsch-Jozsa":
                params["kind"] = st.selectbox("Oráculo", ["balanced","constant1"], key="t3_dj_kind")
                params["n_dj"] = st.number_input("n (entrada ≥1)", min_value=1, value=3, step=1, key="t3_dj_n")
            if algo == "Grover demo":
                params["marked"] = st.text_input("Marcado (binario, n=3)", "111", key="t3_grover_m")
            shots = st.slider("Shots", 100, 8192, 1024, step=100, key="t3_shots")
            run = st.button("Generar + simular", type="primary", use_container_width=True, key="t3_run")

        if run:
            try:
                # crear circuito
                qc = None
                if algo == "Bell":
                    qc = QuantumCircuit(2,2); qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1])
                elif algo == "GHZ":
                    n = int(params["n"]); qc = QuantumCircuit(n, n); qc.h(0)
                    for i in range(n-1): qc.cx(i, i+1)
                    qc.measure(range(n), range(n))
                elif algo == "QFT":
                    n = int(params["n"]); qc = QuantumCircuit(n, n)
                    for j in range(n):
                        qc.h(j)
                        for k in range(j+1, n):
                            qc.cp(np.pi/2**(k-j), k, j)
                    for i in range(n//2): qc.swap(i, n-1-i)
                    qc.measure(range(n), range(n))
                elif algo == "Deutsch-Jozsa":
                    n = int(params["n_dj"])
                    oracle = QuantumCircuit(n+1)
                    if params["kind"]=="balanced":
                        bmask = 1
                        sbin = format(bmask, f"0{n}b")
                        for i,c in enumerate(sbin):
                            if c=="1": oracle.x(i)
                        for i in range(n): oracle.cx(i, n)
                        for i,c in enumerate(sbin):
                            if c=="1": oracle.x(i)
                    else:
                        oracle.x(n)
                    qc = QuantumCircuit(n+1, n)
                    qc.x(n); qc.h(n); qc.h(range(n))
                    qc.append(oracle.to_gate(label=f"Oracle({params['kind']})"), range(n+1))
                    qc.h(range(n)); qc.measure(range(n), range(n))
                elif algo == "Bernstein-Vazirani":
                    s = params["s"]
                    if not all(c in "01" for c in s):
                        st.warning("s debe ser binario."); st.stop()
                    n = len(s)
                    qc = QuantumCircuit(n+1, n)
                    qc.x(n); qc.h(range(n+1)); qc.barrier()
                    for i,bit in enumerate(reversed(s)):
                        if bit=="1": qc.cx(i, n)
                    qc.barrier(); qc.h(range(n)); qc.measure(range(n), range(n))
                elif algo == "Grover demo":
                    marked = params["marked"]
                    if len(marked) != 3 or not all(c in "01" for c in marked):
                        st.warning("Marcado debe ser binario de longitud 3."); st.stop()
                    qc = QuantumCircuit(3,3); qc.h(range(3))
                    oracle = QuantumCircuit(3, name="Oracle")
                    for i,c in enumerate(reversed(marked)):
                        if c=="0": oracle.x(i)
                    oracle.h(2); oracle.mcx([0,1], 2); oracle.h(2)
                    for i,c in enumerate(reversed(marked)):
                        if c=="0": oracle.x(i)
                    diffuser = QuantumCircuit(3, name="Diffuser")
                    diffuser.h(range(3)); diffuser.x(range(3))
                    diffuser.h(2); diffuser.mcx([0,1], 2); diffuser.h(2)
                    diffuser.x(range(3)); diffuser.h(range(3))
                    qc.append(oracle.to_gate(), range(3))
                    qc.append(diffuser.to_gate(), range(3))
                    qc.measure_all()
                elif algo == "Shor demo N=15":
                    def c_amod15(a, power):
                        U = QuantumCircuit(4)
                        for _ in range(power):
                            if a in [2,13]: U.swap(0,1); U.swap(1,2); U.swap(2,3)
                            if a in [7,8]: U.swap(2,3); U.swap(1,2); U.swap(0,1)
                            if a == 11: U.swap(1,3); U.swap(0,2)
                            if a in [7,11,13]:
                                for q in range(4): U.x(q)
                        return U.to_gate(label=f"{a}^k mod 15").control()
                    a = 2
                    n_count = 8
                    qc = QuantumCircuit(n_count+4, n_count)
                    for q in range(n_count): qc.h(q)
                    qc.x(n_count+3)
                    for q in range(n_count):
                        qc.append(c_amod15(a, 2**q), [q] + list(range(n_count, n_count+4)))
                    for j in range(n_count//2): qc.swap(j, n_count-1-j)
                    for j in range(n_count):
                        for m in range(j):
                            qc.cp(-np.pi/2**(j-m), m, j)
                        qc.h(j)
                    qc.measure(range(n_count), range(n_count))

                # Diagrama (mpl si disponible, si no ASCII)
                try:
                    if _HAS_PYLATEXENC:
                        figC = circuit_drawer(qc, output="mpl", style={'name':'mpl'})
                        figC.set_size_inches(9, 3.2)
                        right.pyplot(figC, use_container_width=True)
                        right.download_button("⬇️ Diagrama (PNG)", data=fig_to_png_bytes(figC), file_name="circuit.png", mime="image/png")
                        plt.close(figC)
                    else:
                        right.info("Mostrando diagrama en texto (instala 'pylatexenc' para gráficos).")
                        right.code(str(qc.draw(output='text')), language="text")
                except Exception as e:
                    right.warning("No se pudo dibujar el circuito.")
                    right.exception(e)

                # Simulación (qasm)
                qasm = b["qasm"]
                res = qasm.run(transpile(qc, qasm), shots=int(shots)).result()
                counts = res.get_counts()
                figH = plot_histogram(counts)
                figH.set_size_inches(8.0, 3.0)
                right.pyplot(figH, use_container_width=True)
                right.download_button("⬇️ Histograma (PNG)", data=fig_to_png_bytes(figH), file_name="hist.png", mime="image/png")
                plt.close(figH)

            except Exception as e:
                st.error("Error al generar/simular.")
                st.exception(e)

    # ---------- TAB 4: Código ----------
    with tab4:
        st.subheader("Ejecutor de código Qiskit (define una variable `qc`)")
        code = st.text_area("Código Python", height=240, value=
"""from qiskit import QuantumCircuit
# Ejemplo: GHZ de 3 qubits
qc = QuantumCircuit(3,3)
qc.h(0); qc.cx(0,1); qc.cx(0,2); qc.measure(range(3),range(3))
""")
        if st.button("Ejecutar y simular", type="primary", key="t4_run"):
            ns = {}
            try:
                exec(code, {"np":np, "QuantumCircuit":QuantumCircuit}, ns)
                qc = ns.get("qc", None)
                if qc is None:
                    st.warning("Tu código no definió `qc`."); st.stop()

                # Diagrama
                try:
                    if _HAS_PYLATEXENC:
                        figC = circuit_drawer(qc, output="mpl", style={'name':'mpl'})
                        figC.set_size_inches(9, 3.2)
                        st.pyplot(figC, use_container_width=True)
                        st.download_button("⬇️ Diagrama (PNG)", data=fig_to_png_bytes(figC), file_name="circuit_exec.png", mime="image/png")
                        plt.close(figC)
                    else:
                        st.info("Mostrando diagrama en texto (instala 'pylatexenc' para gráficos).")
                        st.code(str(qc.draw(output='text')), language="text")
                except Exception as e:
                    st.warning("No se pudo dibujar el circuito.")
                    st.exception(e)

                # Simulación (statevector + qasm si hay medidas)
                b = get_backends()
                sim = b["aer"]
                qcsv = qc.remove_final_measurements(inplace=False)
                qcsv.save_statevector()
                sv = sim.run(transpile(qcsv, sim)).result().get_statevector()
                st.markdown("**Vector de estado (sin medidas)**")
                st.code(str(sv), language="text")

                if qc.num_clbits > 0:
                    qasm = b["qasm"]
                    counts = qasm.run(transpile(qc, qasm), shots=1024).result().get_counts()
                    figH = plot_histogram(counts)
                    figH.set_size_inches(8.0, 3.0)
                    st.pyplot(figH, use_container_width=True)
                    st.download_button("⬇️ Histograma (PNG)", data=fig_to_png_bytes(figH), file_name="hist_exec.png", mime="image/png")
                    plt.close(figH)

            except Exception as e:
                st.error("Error al ejecutar tu código.")
                st.exception(e)


# Ejecuta al importar (para tu secure_app.py) y también como script directo
run_app()

if __name__ == "__main__":
    run_app()
