# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for DOL Analyser standalone build.

Build command (run from repo root):
    pyinstaller dol_analyser.spec

Output: dist/DOL_Analyser/   (onedir — zip this folder for distribution)

Tested with:
    PyInstaller >= 6.0
    Python 3.11
    Streamlit >= 1.35
"""
import pathlib
import streamlit

# ── Paths ────────────────────────────────────────────────────────────────────
REPO   = pathlib.Path(SPECPATH)           # repo root (same dir as this file)
ST_DIR = pathlib.Path(streamlit.__file__).parent  # streamlit package root

# ── Analysis ─────────────────────────────────────────────────────────────────
a = Analysis(
    [str(REPO / "launcher.py")],

    datas=[
        # Streamlit's React frontend + all static assets
        (str(ST_DIR),                "streamlit"),
        # Application source
        (str(REPO / "app.py"),       "."),
        (str(REPO / "pipeline"),     "pipeline"),
        (str(REPO / "reports"),      "reports"),
        # Streamlit server config (maxUploadSize = 500)
        (str(REPO / ".streamlit"),   ".streamlit"),
    ],

    hiddenimports=[
        # Streamlit internals
        "streamlit",
        "streamlit.web",
        "streamlit.web.cli",
        "streamlit.web.server",
        "streamlit.runtime",
        "streamlit.runtime.scriptrunner",
        "streamlit.components.v1",
        # Data stack
        "pandas",
        "numpy",
        "scipy",
        "scipy.special._cdflib",
        "scipy.special._ufuncs_cxx",
        "scipy.linalg.cython_blas",
        "scipy.linalg.cython_lapack",
        "sklearn",
        "sklearn.utils._cython_blas",
        "sklearn.utils._weight_vector",
        "sklearn.utils._seq_dataset",
        "sklearn.neighbors._partition_nodes",
        "sklearn.tree._utils",
        "plotly",
        "plotly.io",
        "plotly.express",
        "pyarrow",
        "altair",
        # Streamlit server dependencies
        "tornado",
        "tornado.platform.asyncio",
        "tornado.escape",
        "click",
        "watchdog",
        "watchdog.observers",
        "validators",
        "requests",
        "certifi",
        "urllib3",
        "charset_normalizer",
        # stdlib extras that get missed
        "sqlite3",
        "json",
        "pathlib",
        "datetime",
        "typing_extensions",
        "packaging",
    ],

    excludes=[
        "tkinter", "matplotlib", "IPython",
        "notebook", "jupyter", "pytest",
    ],

    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DOL_Analyser",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no black terminal window on Windows/Mac
    # icon="assets/icon.ico",  # uncomment when you have an icon file
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DOL_Analyser",    # → dist/DOL_Analyser/
)
