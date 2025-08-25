# app_web2.py — versión adaptada para render estable (sin Plotly)
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import streamlit as st

# Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from qiskit.visualization import (
    plot_bloch_multivector,
    plot_histogram,
    circuit_drawer,
)

# ================= Utilidades básicas =================
@st.cache_resource(show_spinner=False)
def get_backends():
    return {
        "aer": Aer.get_backend("aer_simulator"),
        "qasm": Aer.get_backend("qasm_simulator"),
    }

@st.cache_data(show_spinner=False)
def fig_to_png(_fig: Figure, dpi=300):  # <- OJO: _fig con guion bajo
    buf = io.BytesIO()
    _fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    buf.seek(0)
    return buf

def ket_from_angles(theta_deg: float, phi_deg: float):
    th = np.deg2rad(theta_deg); ph = np.deg2rad(phi_deg)
    a = np.cos(th/2.0); b = np.exp(1j*ph)*np.sin(th/2.0)
    v = np.array([a, b], dtype=complex)
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)

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

# ================= App principal =================
def run_app():
    st.title("⚛️ Quantum Toolkit — Qiskit (estable sin Plotly)")

    tab1, tab2, tab3, tab4 = st.tabs([
        "1) Bloch 1-Qubit",
        "2) Puertas 1-Qubit",
        "3) Circuitos (Dashboard)",
        "4) Código Qiskit",
    ])

    # ---------- TAB 1: Bloch ----------
    with tab1:
        st.subheader("Esfera de Bloch (Qiskit + Matplotlib)")
        c1, c2 = st.columns(2)
        with c1:
            th = st.slider("θ (deg)", 0.0, 180.0, 45.0, 0.1, key="t1_theta")
            ph = st.slider("φ (deg)", 0.0, 360.0, 30.0, 0.1, key="t1_phi")

        sv = Statevector(ket_from_angles(th, ph))
        try:
            fig = plot_bloch_multivector(sv)
            fig.set_size_inches(5.6, 5.6)
            st.pyplot(fig, use_container_width=False)
            st.download_button("⬇️ PNG (Bloch)", data=fig_to_png(fig), file_name="bloch.png", mime="image/png")
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
        st.subheader("Puertas 1-Qubit: |ψ_in⟩ → |ψ_out⟩ (Qiskit)")
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
                fig_in = plot_bloch_multivector(sv_in); fig_in.set_size_inches(5.0,5.0)
                cols[0].pyplot(fig_in, use_container_width=True); plt.close(fig_in)
            except Exception as e:
                cols[0].error("Error Bloch IN"); cols[0].exception(e)
            try:
                fig_out = plot_bloch_multivector(sv_out); fig_out.set_size_inches(5.0,5.0)
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

                # Diagrama (mpl)
                try:
                    figC = circuit_drawer(qc, output="mpl", style={'name':'mpl'})
                    figC.set_size_inches(9, 3.2)
                    right.pyplot(figC, use_container_width=True)
                    right.download_button("⬇️ Diagrama (PNG)", data=fig_to_png(figC), file_name="circuit.png", mime="image/png")
                    plt.close(figC)
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
                right.download_button("⬇️ Histograma (PNG)", data=fig_to_png(figH), file_name="hist.png", mime="image/png")
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
                    figC = circuit_drawer(qc, output="mpl", style={'name':'mpl'})
                    figC.set_size_inches(9, 3.2)
                    st.pyplot(figC, use_container_width=True)
                    st.download_button("⬇️ Diagrama (PNG)", data=fig_to_png(figC), file_name="circuit_exec.png", mime="image/png")
                    plt.close(figC)
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
                    st.download_button("⬇️ Histograma (PNG)", data=fig_to_png(figH), file_name="hist_exec.png", mime="image/png")
                    plt.close(figH)

            except Exception as e:
                st.error("Error al ejecutar tu código.")
                st.exception(e)
