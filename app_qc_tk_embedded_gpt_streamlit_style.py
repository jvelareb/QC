# -*- coding: utf-8 -*-
"""
QC Tkinter • Embedded RAG • GPT or Extractive • Streamlit-style Circuits
- PDFs embebidos en assets/pdfs/ y RAG precalculado (prebuild_rag.py) en assets/index/.
- Chat siempre usa ese RAG embebido:
    * Si OPENAI_API_KEY -> GPT con contexto.
    * Si no hay clave -> modo extractivo (muestra pasajes top).
- Pestañas:
    • Bloch Sphere (1-qubit).
    • Circuits con estética tipo Streamlit:
        - Encabezado y controles limpios
        - Tarjetas de métricas: Qubits, Depth, Size, Ops, Classical, Opt. depth
        - Subpestañas: Circuit / Histogram / Q-sphere / Statevector
"""

import os, io, math, tempfile, tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from numpy import pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg

# ---- Qiskit (opcional para simulación) ----
QISKIT_OK = True
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    try:
        from qiskit_aer import Aer
        AER_OK = True
    except Exception:
        AER_OK = False
    from qiskit.visualization import circuit_drawer, plot_histogram, plot_state_qsphere
except Exception:
    QISKIT_OK, AER_OK = False, False

# ---- RAG persistente (cargado, no construido) ----
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# ---- OpenAI (opcional) ----
OPENAI_OK = True
try:
    from openai import OpenAI
except Exception:
    OPENAI_OK = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_INDEX_DIR = os.path.join(APP_DIR, "assets", "index")
INDEX_NPZ = os.path.join(ASSETS_INDEX_DIR, "rag_tfidf.npz")
INDEX_TSV = os.path.join(ASSETS_INDEX_DIR, "rag_meta.tsv")

# ---------- Utilidades QC ----------
def pretty_c(z: complex, nd=3):
    a = np.round(z.real, nd); b = np.round(z.imag, nd)
    if abs(b) < 10**(-nd): return f"{a:.{nd}f}"
    if abs(a) < 10**(-nd): return f"{b:.{nd}f}i"
    sign = "+" if b >= 0 else "-"
    return f"{a:.{nd}f}{sign}{abs(b):.{nd}f}i"

def state_from_angles(theta, phi):
    a = math.cos(theta / 2.0)
    b = np.exp(1j * phi) * math.sin(theta / 2.0)
    v = np.array([[a], [b]], dtype=complex)
    return v / np.linalg.norm(v)

def U_gate(name, param=None):
    I = np.array([[1,0],[0,1]], dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
    S = np.array([[1,0],[0,1j]], dtype=complex)
    T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]], dtype=complex)
    if name == "I": return I
    if name == "X": return X
    if name == "Y": return Y
    if name == "Z": return Z
    if name == "H": return H
    if name == "S": return S
    if name == "T": return T
    if name == "RX":
        th = 0.0 if param is None else float(param)
        return np.array([[np.cos(th/2), -1j*np.sin(th/2)],
                         [-1j*np.sin(th/2), np.cos(th/2)]], dtype=complex)
    if name == "RY":
        th = 0.0 if param is None else float(param)
        return np.array([[np.cos(th/2), -np.sin(th/2)],
                         [np.sin(th/2),  np.cos(th/2)]], dtype=complex)
    if name == "RZ":
        th = 0.0 if param is None else float(param)
        return np.array([[np.exp(-1j*th/2), 0],
                         [0, np.exp(1j*th/2)]], dtype=complex)
    return I

def bloch_xyz(psi_col):
    a, b = psi_col[0,0], psi_col[1,0]
    x = 2*np.real(np.conj(a)*b); y = 2*np.imag(np.conj(b)*a); z = (abs(a)**2) - (abs(b)**2)
    return float(x), float(y), float(z)

def imread_bytes(buf: io.BytesIO):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(buf.read()); tmp.flush(); tmp.close()
    arr = mpimg.imread(tmp.name)
    try: os.unlink(tmp.name)
    except: pass
    return arr

# ---------- RAG loader (solo carga; índice ya existe) ----------
class PDFIndexTFIDF:
    def __init__(self):
        self.chunks = []
        self.vectorizer = None
        self.matrix = None
        self.is_built = False

    def load(self, index_npz=INDEX_NPZ, meta_tsv=INDEX_TSV):
        if not (os.path.exists(index_npz) and os.path.exists(meta_tsv)):
            raise RuntimeError(
                f"No se encontró el índice RAG.\nEsperado en:\n  {index_npz}\n  {meta_tsv}\n"
                "Ejecuta primero: python prebuild_rag.py"
            )
        data = np.load(index_npz, allow_pickle=True)
        indptr, indices, vals = data["indptr"], data["indices"], data["data"]
        self.matrix = csr_matrix((vals, indices, indptr))
        vocab_tokens, vocab_ids = list(data["vocab_tokens"]), list(data["vocab_ids"])
        vocab = {tok: int(i) for tok, i in zip(vocab_tokens, vocab_ids)}
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectorizer.vocabulary_ = vocab
        self.vectorizer.fixed_vocabulary_ = True
        if "idf" in data and data["idf"] is not None:
            self.vectorizer.idf_ = data["idf"]
        # chunks
        self.chunks = []
        with open(meta_tsv, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    src, txt = line.rstrip("\n").split("\t", 1)
                except ValueError:
                    continue
                self.chunks.append({"src": src, "text": txt})
        self.is_built = True

    def query(self, question, k=5):
        qv = self.vectorizer.transform([question])
        sims = cosine_similarity(qv, self.matrix)[0]
        idx = np.argsort(-sims)[:k]
        return [{"text": self.chunks[i]["text"], "src": self.chunks[i]["src"], "score": float(sims[i])} for i in idx]

# ---------- GPT backend (o fallback extractivo) ----------
class GPTorExtractive:
    def __init__(self, temperature=0.2, model="gpt-4o-mini"):
        self.temperature = float(temperature)
        self.model = model
        self.has_gpt = OPENAI_OK and bool(os.getenv("OPENAI_API_KEY"))
        self.client = OpenAI() if self.has_gpt else None

    def answer(self, question, contexts):
        if self.has_gpt:
            sys = "You are a helpful assistant. Prefer the provided CONTEXT if relevant. Cite [filename] when using context."
            msgs = [{"role":"system","content":sys}]
            if contexts:
                joined = "\n\n".join([f"[{c['src']} score={c['score']:.3f}]\n{c['text']}" for c in contexts])
                msgs.append({"role":"system","content":f"CONTEXT:\n{joined}"})
            msgs.append({"role":"user","content":question})
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, temperature=self.temperature
            )
            return resp.choices[0].message.content
        # Fallback sin GPT: muestra pasajes top
        if contexts:
            return "\n\n".join([f"[{c['src']}]\n{c['text']}" for c in contexts])
        return "Sin OPENAI_API_KEY. Agrega tu clave para obtener respuesta generativa."

# ---------- Chat panel (siempre RAG embebido) ----------
class ChatPanel(ttk.Frame):
    def __init__(self, parent, title="Chat"):
        super().__init__(parent)
        self.index = PDFIndexTFIDF()
        self.llm = GPTorExtractive(temperature=0.2, model="gpt-4o-mini")

        ttk.Label(self, text=title, font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")

        self.info = tk.Text(self, width=46, height=5, font=("Consolas", 9))
        self.info.grid(row=1, column=0, sticky="ew", pady=(2,4))
        self._init_rag()

        ttk.Label(self, text="Question:").grid(row=2, column=0, sticky="w")
        self.qbox = tk.Text(self, width=46, height=4, font=("Consolas", 9))
        self.qbox.grid(row=3, column=0, sticky="ew")

        ttk.Button(self, text="Ask", command=self._ask).grid(row=4, column=0, sticky="w", pady=(4,2))

        ttk.Label(self, text="Answer:").grid(row=5, column=0, sticky="w")
        self.abox = tk.Text(self, width=46, height=18, font=("Consolas", 9))
        self.abox.grid(row=6, column=0, sticky="ew")

    def _println(self, s):
        self.info.insert("end", s + "\n"); self.info.see("end")

    def _init_rag(self):
        try:
            self.index.load(INDEX_NPZ, INDEX_TSV)
            mode = "GPT" if self.llm.has_gpt else "extractivo (sin GPT)"
            self._println(f"[RAG] Índice cargado (chunks={len(self.index.chunks)}). Modo: {mode}")
        except Exception as e:
            self._println(f"[ERROR] {e}")

    def _ask(self):
        q = self.qbox.get("1.0","end").strip()
        if not q: return
        try:
            ctxs = self.index.query(q, k=5)
        except Exception:
            ctxs = None
        ans = self.llm.answer(q, ctxs)
        self.abox.insert("end", ans + "\n\n"); self.abox.see("end")

# ---------- Bloch Tab ----------
class BlochTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent); self._build()

    def _build(self):
        self.columnconfigure(0, weight=5); self.columnconfigure(1, weight=2); self.rowconfigure(0, weight=1)
        left = ttk.Frame(self, padding=8); left.grid(row=0, column=0, sticky="nsew")
        right = ChatPanel(self, title="Chat (Bloch)"); right.grid(row=0, column=1, sticky="nsew", padx=(6,0))

        # Encabezado
        ttk.Label(left, text="Bloch Sphere (1 qubit)", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0,6))

        # Controles compactos como Streamlit
        frm = ttk.LabelFrame(left, text="Controls"); frm.grid(row=1, column=0, sticky="ew")
        for c in range(4): frm.columnconfigure(c, weight=1)
        ttk.Label(frm, text="Units").grid(row=0, column=0, sticky="e")
        self.units = ttk.Combobox(frm, values=["deg","rad"], state="readonly", width=6); self.units.set("deg")
        self.units.grid(row=0, column=1, sticky="w")
        ttk.Label(frm, text="θ").grid(row=0, column=2, sticky="e")
        self.theta = tk.DoubleVar(value=60.0); ttk.Entry(frm, textvariable=self.theta, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(frm, text="φ").grid(row=1, column=0, sticky="e")
        self.phi = tk.DoubleVar(value=90.0); ttk.Entry(frm, textvariable=self.phi, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(frm, text="Gate").grid(row=1, column=2, sticky="e")
        self.gate = ttk.Combobox(frm, values=["I","X","Y","Z","H","S","T","RX","RY","RZ"], state="readonly", width=8); self.gate.set("I")
        self.gate.grid(row=1, column=3, sticky="w")
        ttk.Label(frm, text="Angle (RX/RY/RZ)").grid(row=2, column=0, sticky="e")
        self.ang = tk.DoubleVar(value=0.0); ttk.Entry(frm, textvariable=self.ang, width=8).grid(row=2, column=1, sticky="w")
        ttk.Button(frm, text="Draw", command=self.draw).grid(row=2, column=3, sticky="e")

        # Figura
        self.fig = Figure(figsize=(6.4,5.0), dpi=100); self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left); self.canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew", pady=(6,6))
        left.rowconfigure(2, weight=1)
        self.out = tk.Text(left, height=6, font=("Consolas",9)); self.out.grid(row=3, column=0, sticky="ew")

        self._init_sphere(); self.draw()

    def _init_sphere(self):
        self.ax.clear()
        u = np.linspace(0, 2*np.pi, 60); v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v)); y = np.outer(np.sin(u), np.sin(v)); z = np.outer(np.ones_like(u), np.cos(v))
        self.ax.plot_wireframe(x, y, z, rstride=3, cstride=3, alpha=0.18, linewidth=0.6)
        L = 1.1
        self.ax.plot([-L,L],[0,0],[0,0], color="k"); self.ax.plot([0,0],[-L,L],[0,0], color="k"); self.ax.plot([0,0],[0,0],[-L,L], color="k")
        self.ax.set_xlim([-1.2,1.2]); self.ax.set_ylim([-1.2,1.2]); self.ax.set_zlim([-1.2,1.2]); self.ax.set_box_aspect([1,1,1])
        self.ax.set_title("Bloch Sphere"); self.canvas.draw()

    def draw(self):
        th = float(self.theta.get()); ph = float(self.phi.get())
        if self.units.get()=="deg": th = np.deg2rad(th); ph = np.deg2rad(ph)
        psi = state_from_angles(th, ph); g = self.gate.get()
        if g in ("RX","RY","RZ"):
            ang = float(self.ang.get()); 
            if self.units.get()=="deg": ang = np.deg2rad(ang)
            U = U_gate(g, ang); gdesc = f"{g}({np.rad2deg(ang):.1f}°)"
        else:
            U = U_gate(g); gdesc = g
        psi_out = U @ psi
        x,y,z = bloch_xyz(psi_out)
        self._init_sphere(); self.ax.quiver(0,0,0, x,y,z, length=1.0, normalize=False, color="crimson", linewidth=2); self.canvas.draw()
        p0 = 100*abs(psi_out[0,0])**2; p1 = 100*abs(psi_out[1,0])**2
        self.out.delete("1.0","end")
        self.out.insert("end", f"Gate: {gdesc}\nU = [[{pretty_c(U[0,0])}, {pretty_c(U[0,1])}], [{pretty_c(U[1,0])}, {pretty_c(U[1,1])}]]\n")
        self.out.insert("end", f"|ψ_out> = [{pretty_c(psi_out[0,0])}, {pretty_c(psi_out[1,0])}]^T\nP(0)={p0:.2f}%, P(1)={p1:.2f}%\n")

# ---------- Circuits Tab (streamlit-style) ----------
class MetricCard(ttk.Frame):
    def __init__(self, parent, title, var_str):
        super().__init__(parent, padding=(10,8,10,8))
        self.configure(borderwidth=1, relief="groove")
        self.title = ttk.Label(self, text=title, font=("Segoe UI", 9))
        self.value = ttk.Label(self, textvariable=var_str, font=("Segoe UI", 14, "bold"))
        self.title.grid(row=0, column=0, sticky="w")
        self.value.grid(row=1, column=0, sticky="w")

class CircuitsTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent); self.ops = []; self._build()

    def _build(self):
        self.columnconfigure(0, weight=5); self.columnconfigure(1, weight=2); self.rowconfigure(0, weight=1)
        left = ttk.Frame(self, padding=8); left.grid(row=0, column=0, sticky="nsew")
        right = ChatPanel(self, title="Chat (Circuits)"); right.grid(row=0, column=1, sticky="nsew", padx=(6,0))

        # Encabezado tipo Streamlit
        ttk.Label(left, text="Quantum Circuits", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(left, text="Build, simulate and visualize. Metrics update after each simulation.",
                  font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=(0,8))

        # Controles "limpios"
        bld = ttk.LabelFrame(left, text="Controls"); bld.grid(row=2, column=0, sticky="ew")
        for c in range(8): bld.columnconfigure(c, weight=1)
        if not QISKIT_OK:
            ttk.Label(bld, text="Qiskit not installed. pip install qiskit qiskit-aer", foreground="darkred").grid(
                row=0, column=0, columnspan=8, sticky="w"
            )

        ttk.Label(bld, text="Qubits").grid(row=1, column=0, sticky="e")
        self.nq = tk.IntVar(value=2); ttk.Spinbox(bld, from_=1, to=6, width=5, textvariable=self.nq).grid(row=1, column=1, sticky="w")
        ttk.Label(bld, text="Shots").grid(row=1, column=2, sticky="e")
        self.shots = tk.IntVar(value=1024); ttk.Spinbox(bld, from_=1, to=50000, width=7, textvariable=self.shots).grid(row=1, column=3, sticky="w")

        self.gate = ttk.Combobox(bld, values=["H","X","Y","Z","S","T","RX","RY","RZ","CX","CZ","SWAP","MEASURE"],
                                 state="readonly", width=10)
        self.gate.set("H"); self.gate.grid(row=2, column=0, sticky="w")
        ttk.Label(bld, text="target").grid(row=2, column=1, sticky="e")
        self.tgt = tk.IntVar(value=0); ttk.Spinbox(bld, from_=0, to=9, width=5, textvariable=self.tgt).grid(row=2, column=2, sticky="w")
        ttk.Label(bld, text="control").grid(row=2, column=3, sticky="e")
        self.ctrl = tk.IntVar(value=0); ttk.Spinbox(bld, from_=0, to=9, width=5, textvariable=self.ctrl).grid(row=2, column=4, sticky="w")
        ttk.Label(bld, text="angle (deg)").grid(row=2, column=5, sticky="e")
        self.ang = tk.DoubleVar(value=90.0); ttk.Entry(bld, textvariable=self.ang, width=7).grid(row=2, column=6, sticky="w")

        cmd = ttk.Frame(bld); cmd.grid(row=3, column=0, columnspan=8, sticky="w", pady=(4,2))
        ttk.Button(cmd, text="Add", command=self._add).grid(row=0, column=0, padx=2)
        ttk.Button(cmd, text="Undo", command=self._undo).grid(row=0, column=1, padx=2)
        ttk.Button(cmd, text="Clear", command=self._clear).grid(row=0, column=2, padx=2)
        ttk.Button(cmd, text="Simulate", command=self._simulate).grid(row=0, column=3, padx=8)

        # Secuencia
        ttk.Label(left, text="Sequence").grid(row=3, column=0, sticky="w", pady=(6,0))
        self.listbox = tk.Listbox(left, width=80, height=6); self.listbox.grid(row=4, column=0, sticky="ew")

        # Métricas tipo tarjetas (fila de 6)
        grid_metrics = ttk.Frame(left); grid_metrics.grid(row=5, column=0, sticky="ew", pady=(8,2))
        for c in range(6): grid_metrics.columnconfigure(c, weight=1)
        self.met_qubits = tk.StringVar(value="-")
        self.met_depth  = tk.StringVar(value="-")
        self.met_size   = tk.StringVar(value="-")
        self.met_ops    = tk.StringVar(value="-")
        self.met_class  = tk.StringVar(value="-")
        self.met_optdep = tk.StringVar(value="-")
        MetricCard(grid_metrics, "Qubits", self.met_qubits).grid(row=0, column=0, sticky="ew", padx=3)
        MetricCard(grid_metrics, "Depth",  self.met_depth ).grid(row=0, column=1, sticky="ew", padx=3)
        MetricCard(grid_metrics, "Size",   self.met_size  ).grid(row=0, column=2, sticky="ew", padx=3)
        MetricCard(grid_metrics, "Ops",    self.met_ops   ).grid(row=0, column=3, sticky="ew", padx=3)
        MetricCard(grid_metrics, "Classical", self.met_class).grid(row=0, column=4, sticky="ew", padx=3)
        MetricCard(grid_metrics, "Opt. depth", self.met_optdep).grid(row=0, column=5, sticky="ew", padx=3)

        # Notebook de visualizaciones
        figs = ttk.Notebook(left); figs.grid(row=6, column=0, sticky="nsew", pady=(6,0))
        left.rowconfigure(6, weight=1)

        frm_c = ttk.Frame(figs); frm_h = ttk.Frame(figs); frm_q = ttk.Frame(figs); frm_s = ttk.Frame(figs)
        figs.add(frm_c, text="Circuit"); figs.add(frm_h, text="Histogram"); figs.add(frm_q, text="Q-sphere"); figs.add(frm_s, text="Statevector")

        self.fig_c = Figure(figsize=(7.6,3.0), dpi=100); self.ax_c = self.fig_c.add_subplot(111); self.ax_c.axis("off")
        self.can_c = FigureCanvasTkAgg(self.fig_c, master=frm_c); self.can_c.get_tk_widget().pack(fill="both", expand=True)

        self.fig_h = Figure(figsize=(7.6,3.0), dpi=100); self.ax_h = self.fig_h.add_subplot(111)
        self.can_h = FigureCanvasTkAgg(self.fig_h, master=frm_h); self.can_h.get_tk_widget().pack(fill="both", expand=True)

        self.fig_q = Figure(figsize=(6.2,6.2), dpi=100); self.ax_q = self.fig_q.add_subplot(111)
        self.can_q = FigureCanvasTkAgg(self.fig_q, master=frm_q); self.can_q.get_tk_widget().pack(fill="both", expand=True)

        self.fig_s = Figure(figsize=(7.6,3.0), dpi=100); self.ax_s = self.fig_s.add_subplot(111)
        self.can_s = FigureCanvasTkAgg(self.fig_s, master=frm_s); self.can_s.get_tk_widget().pack(fill="both", expand=True)
        self.sv_text = tk.Text(frm_s, height=6, font=("Consolas",9)); self.sv_text.pack(fill="x")

        # Consola (cálculos y resumen como en Streamlit)
        ttk.Label(left, text="Calculations").grid(row=7, column=0, sticky="w", pady=(6,0))
        self.console = tk.Text(left, height=8, font=("Consolas",9)); self.console.grid(row=8, column=0, sticky="ew")

    def _log(self, s): self.console.insert("end", str(s)+"\n"); self.console.see("end")
    def _add(self):
        op = {"g": self.gate.get(), "t": int(self.tgt.get()), "c": int(self.ctrl.get()), "ang": float(self.ang.get())}
        self.ops.append(op); self.listbox.insert("end", self._op_str(op))
    def _undo(self):
        if self.ops: self.ops.pop(); self.listbox.delete("end")
    def _clear(self): self.ops.clear(); self.listbox.delete(0,"end")
    def _op_str(self, op):
        g=op["g"]
        if g in ("RX","RY","RZ"): return f"{g}({op['ang']} deg) q[{op['t']}]"
        if g in ("CX","CZ","SWAP"): return f"{g} q[{op['c']}]→q[{op['t']}]"
        if g=="MEASURE": return "MEASURE (all)"
        return f"{g} q[{op['t']}]"

    def _build_qc(self):
        nq = int(self.nq.get()); qc = QuantumCircuit(nq, nq)
        for op in self.ops:
            g,t,c,ang = op["g"], op["t"], op["c"], op["ang"]
            if t>=nq or c>=nq: raise ValueError(f"Qubit index out of range in {self._op_str(op)}")
            if g=="H": qc.h(t)
            elif g=="X": qc.x(t)
            elif g=="Y": qc.y(t)
            elif g=="Z": qc.z(t)
            elif g=="S": qc.s(t)
            elif g=="T": qc.t(t)
            elif g=="RX": qc.rx(np.deg2rad(ang), t)
            elif g=="RY": qc.ry(np.deg2rad(ang), t)
            elif g=="RZ": qc.rz(np.deg2rad(ang), t)
            elif g=="CX": qc.cx(c,t)
            elif g=="CZ": qc.cz(c,t)
            elif g=="SWAP": qc.swap(c,t)
            elif g=="MEASURE": qc.measure(range(nq), range(nq))
        return qc

    def _update_metrics(self, qc):
        try:
            ops = qc.count_ops()
            self.met_qubits.set(str(qc.num_qubits))
            self.met_class.set(str(qc.num_clbits))
            self.met_depth.set(str(qc.depth()))
            self.met_size.set(str(qc.size()))
            self.met_ops.set(str(int(sum(ops.values()))))
            # depth optimizada (como Streamlit cuando pones opt level alto)
            try:
                qct = transpile(qc, optimization_level=3)
                self.met_optdep.set(str(qct.depth()))
            except Exception:
                self.met_optdep.set("-")
        except Exception:
            self.met_qubits.set("-"); self.met_class.set("-"); self.met_depth.set("-")
            self.met_size.set("-"); self.met_ops.set("-"); self.met_optdep.set("-")

    def _simulate(self):
        if not QISKIT_OK:
            messagebox.showerror("Qiskit", "Install qiskit y qiskit-aer para simular."); return
        try:
            qc = self._build_qc()
        except Exception as e:
            self._log(f"[Build error] {e}"); return

        # Métricas arriba
        self._update_metrics(qc)

        # 1) Circuit
        self.ax_c.clear()
        try:
            fig = circuit_drawer(qc, output="mpl", fold=-1)
            buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=140); buf.seek(0)
            self.ax_c.imshow(imread_bytes(buf)); self.ax_c.axis("off")
        except Exception as e:
            self.ax_c.text(0.5,0.5,f"Draw error: {e}", ha="center", va="center")
        self.can_c.draw()

        # 2) Statevector
        self.ax_s.clear(); self.sv_text.delete("1.0","end")
        probs_dict = {}
        try:
            base = qc.remove_final_measurements(inplace=False); base.save_statevector()
            if AER_OK:
                sim = Aer.get_backend("aer_simulator")
                sv = sim.run(transpile(base, sim)).result().get_statevector()
            else:
                sv = Statevector.from_instruction(base).data
            probs = np.abs(np.asarray(sv))**2
            self.ax_s.bar(range(len(probs)), probs); self.ax_s.set_title("|amp|^2 by basis"); self.can_s.draw()
            self.sv_text.insert("end", str(sv))
            # Probabilidades legibles (como tabla texto tipo Streamlit)
            for idx, p in enumerate(probs):
                if p > 1e-6:
                    probs_dict[format(idx, f"0{qc.num_qubits}b")] = float(p)
        except Exception as e:
            self.ax_s.text(0.5,0.5,f"SV error: {e}", ha="center", va="center"); self.can_s.draw()

        # 3) Histogram (si hay medidas)
        self.ax_h.clear()
        counts = None
        if qc.num_clbits>0 and AER_OK:
            try:
                qasm = Aer.get_backend("qasm_simulator")
                shots = int(self.shots.get())
                counts = qasm.run(transpile(qc, qasm), shots=shots).result().get_counts()
                plot_histogram(counts, ax=self.ax_h); self.ax_h.set_title(f"Histogram ({shots} shots)")
            except Exception as e:
                self.ax_h.text(0.5,0.5,f"Hist error: {e}", ha="center", va="center")
        else:
            self.ax_h.text(0.5,0.5,"No measures or Aer missing.", ha="center", va="center")
        self.can_h.draw()

        # 4) Q-sphere
        self.ax_q.clear()
        try:
            base = qc.remove_final_measurements(inplace=False) if qc.num_clbits>0 else qc
            if AER_OK:
                sim = Aer.get_backend("aer_simulator"); base = base.copy(); base.save_statevector()
                sv = sim.run(transpile(base, sim)).result().get_statevector()
                figQ = plot_state_qsphere(Statevector(sv))
            else:
                sv = Statevector.from_instruction(base); figQ = plot_state_qsphere(sv)
            buf = io.BytesIO(); figQ.savefig(buf, format="png", bbox_inches="tight", dpi=140); buf.seek(0)
            self.ax_q.imshow(imread_bytes(buf)); self.ax_q.axis("off")
        except Exception as e:
            self.ax_q.text(0.5,0.5,f"Q-sphere error: {e}", ha="center", va="center")
        self.can_q.draw()

        # --- Consola con resumen “estilo Streamlit” ---
        self.console.delete("1.0","end")
        self._log("=== Circuit summary ===")
        self._log(f"Depth        : {self.met_depth.get()}")
        self._log(f"Size         : {self.met_size.get()}")
        self._log(f"Ops (total)  : {self.met_ops.get()}")
        self._log(f"Classical bits: {self.met_class.get()}")
        self._log(f"Opt. depth(3): {self.met_optdep.get()}")
        try:
            ops = qc.count_ops()
            self._log("Count ops    : " + ", ".join([f"{k}:{int(v)}" for k,v in ops.items()]))
        except Exception:
            pass
        if probs_dict:
            self._log("Probabilities (|amp|^2):")
            for b, p in sorted(probs_dict.items(), key=lambda x: -x[1]):
                self._log(f"  {b} : {p:.6f}")
        if counts:
            self._log("Counts:")
            for b, n in sorted(counts.items(), key=lambda x: -x[1]):
                self._log(f"  {b} : {n}")

# ---------- App ----------
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QC Tkinter • Embedded RAG • GPT or Extractive (Streamlit-style)")
        self.geometry("1420x880")
        try:
            # Un look algo más limpio
            style = ttk.Style(self)
            if "clam" in style.theme_names():
                style.theme_use("clam")
        except Exception:
            pass
        nb = ttk.Notebook(self); nb.pack(fill="both", expand=True)
        nb.add(BlochTab(nb), text="Bloch Sphere")
        nb.add(CircuitsTab(nb), text="Circuits")

if __name__ == "__main__":
    app = MainApp(); app.mainloop()
