from setuptools import setup, find_packages

setup(
    name="quantum_book",
    version="1.0.0",
    description="Code repository for the Quantum Computing book",
    author="J. Velasco",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "qiskit>=1.0.0",
        "qiskit-aer>=0.14.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "matplotlib>=3.8.0",
        "plotly>=5.20.0",
        "ipywidgets>=8.1.0",
        "streamlit>=1.35.0",
    ],
)
