"""
app.py
Główny interfejs aplikacji Streamlit:
Modelowanie i wizualizacja relacji między parametrami geometrycznymi struktury Gyroidu
a jej macierzą sztywności 6x6 (notacja Voigta) z wykorzystaniem sieci neuronowej PyTorch.
"""

import os
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import importlib
import tpms_generator as tg
import nn_model as nm
import export_3d as exp3d

importlib.reload(tg)
importlib.reload(nm)
importlib.reload(exp3d)

# ==========================================
# Konfiguracja strony Streamlit
# ==========================================
st.set_page_config(
    page_title="Gyroid Metamaterial Stiffness NN Studio",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS dla nowoczesnego wyglądu (glassmorphism, typografia, karty)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        color: #f8fafc;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .main-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(10px);
    }
    
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 0.2rem;
    }
    
    .formula-box {
        background: rgba(15, 23, 42, 0.7);
        border-left: 4px solid #818cf8;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# Inicjalizacja stanu sesji i Caching
# ==========================================
# Inicjalizacja stanu sesji i Caching
# ==========================================
@st.cache_data(show_spinner=False)
def get_cached_tpms_grid(tpms_type: str, nx: float, ny: float, nz: float, t: float, resolution: int, mode: str = "network", wall_thickness: float = 0.20):
    """Generowanie siatki TPMS / Auxetic z pamięcią podręczną."""
    return tg.generate_tpms_grid(tpms_type, nx, ny, nz, t, resolution=resolution, mode=mode, wall_thickness=wall_thickness)


@st.cache_data(show_spinner=False)
def get_cached_gyroid_grid(nx: float, ny: float, nz: float, t: float, resolution: int, mode: str = "network", wall_thickness: float = 0.20):
    """Generowanie siatki Gyroidu z pamięcią podręczną."""
    return get_cached_tpms_grid("gyroid", nx, ny, nz, t, resolution=resolution, mode=mode, wall_thickness=wall_thickness)


@st.cache_data(show_spinner=False)
def get_cached_diamond_grid(nx: float, ny: float, nz: float, t: float, resolution: int, mode: str = "network", wall_thickness: float = 0.20):
    """Generowanie siatki Schwarz Diamond z pamięcią podręczną."""
    return get_cached_tpms_grid("diamond", nx, ny, nz, t, resolution=resolution, mode=mode, wall_thickness=wall_thickness)


@st.cache_data(show_spinner=False)
def get_cached_primitive_grid(nx: float, ny: float, nz: float, t: float, resolution: int, mode: str = "network", wall_thickness: float = 0.20):
    """Generowanie siatki Schwarz Primitive z pamięcią podręczną."""
    return get_cached_tpms_grid("primitive", nx, ny, nz, t, resolution=resolution, mode=mode, wall_thickness=wall_thickness)


@st.cache_data(show_spinner=False)
def get_cached_auxetic_grid(nx: float, ny: float, nz: float, t: float, resolution: int, mode: str = "network", wall_thickness: float = 0.20):
    """Generowanie siatki Re-entrant Auxetic z pamięcią podręczną."""
    return get_cached_tpms_grid("auxetic", nx, ny, nz, t, resolution=resolution, mode=mode, wall_thickness=wall_thickness)


@st.cache_data(show_spinner=False)
def get_cached_neovius_grid(nx: float, ny: float, nz: float, t: float, resolution: int, mode: str = "network", wall_thickness: float = 0.20):
    """Generowanie siatki Schoen Neovius C(P) z pamięcią podręczną."""
    return get_cached_tpms_grid("neovius", nx, ny, nz, t, resolution=resolution, mode=mode, wall_thickness=wall_thickness)


@st.cache_data(show_spinner=False)
def get_cached_iwp_grid(nx: float, ny: float, nz: float, t: float, resolution: int, mode: str = "network", wall_thickness: float = 0.20):
    """Generowanie siatki Schoen I-WP z pamięcią podręczną."""
    return get_cached_tpms_grid("iwp", nx, ny, nz, t, resolution=resolution, mode=mode, wall_thickness=wall_thickness)


@st.cache_data(show_spinner=False)
def get_cached_octet_grid(nx: float, ny: float, nz: float, t: float, resolution: int, mode: str = "network", wall_thickness: float = 0.20):
    """Generowanie siatki Octet-Truss (Schoen F-RD) z pamięcią podręczną."""
    return get_cached_tpms_grid("octet", nx, ny, nz, t, resolution=resolution, mode=mode, wall_thickness=wall_thickness)


@st.cache_data(show_spinner=False)
def get_cached_synthetic_data(num_samples: int, noise_level: float, seed: int, tpms_type: str = "gyroid", mode: str = "network", wall_thickness: float = 0.20):
    """Generowanie danych syntetycznych z pamięcią podręczną."""
    return tg.generate_synthetic_dataset(num_samples=num_samples, noise_level=noise_level, random_seed=seed, tpms_type=tpms_type, mode=mode, wall_thickness=wall_thickness)


@st.cache_data(show_spinner=False)
def get_cached_young_surface(C: np.ndarray, num_theta: int = 40, num_phi: int = 80):
    """Obliczanie kierunkowego modułu Younga z pamięcią podręczną."""
    return tg.calculate_directional_youngs_modulus(C, num_theta=num_theta, num_phi=num_phi)


def get_bounding_box_trace(Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0, color: str = "#334155", width: float = 2.0) -> go.Scatter3d:
    """Tworzy ścieżkę 3D krawędzi obramowania komórki elementarnej (12 krawędzi prostopadłościanu)."""
    x = [0, Lx, Lx, 0, 0, None, 0, Lx, Lx, 0, 0, None, 0, 0, None, Lx, Lx, None, Lx, Lx, None, 0, 0]
    y = [0, 0, Ly, Ly, 0, None, 0, 0, Ly, Ly, 0, None, 0, 0, None, 0, 0, None, Ly, Ly, None, Ly, Ly]
    z = [0, 0, 0, 0, 0, None, Lz, Lz, Lz, Lz, Lz, None, 0, Lz, None, 0, Lz, None, 0, Lz, None, 0, Lz]
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="none",
        showlegend=False,
        name="Krawędzie komórki",
    )


def build_gallery_figure(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    F: np.ndarray,
    colorscale: str,
    opacity: float = 0.85,
    show_edges: bool = True,
    show_bbox: bool = True,
    height: int = 360,
) -> go.Figure:
    """Buduje wykres 3D Isosurface z obsługą przezroczystości, krawędzi konturów i ramki komórki elementarnej."""
    contour_cfg = dict(show=True, color="#0f172a", width=3) if show_edges else dict(show=False)
    fig = go.Figure(
        data=go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=F.flatten(),
            isomin=-0.08,
            isomax=0.08,
            surface_count=2,
            colorscale=colorscale,
            opacity=opacity,
            caps=dict(x_show=False, y_show=False, z_show=False),
            contour=contour_cfg,
            showscale=False,
        )
    )
    if show_bbox:
        fig.add_trace(get_bounding_box_trace(Lx=1.0, Ly=1.0, Lz=1.0, color="#334155", width=2.5))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", nticks=4),
            yaxis=dict(title="Y", nticks=4),
            zaxis=dict(title="Z", nticks=4),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.4)),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=height,
    )
    return fig


def apply_preset():
    """Obsługa zmiany konfiguracji predefiniowanej (Preset)."""
    preset_name = st.session_state.get("preset_selector", "Własna konfiguracja (Custom)")
    if preset_name == "Własna konfiguracja (Custom)":
        return

    preset_configs = {
        "Izotropowy standardowy (nx=2, ny=2, nz=2, t=0)": {
            "nx": 2.0, "ny": 2.0, "nz": 2.0, "t": 0.0,
        },
        "Anizotropowy belkowy (nx=4.5, ny=1.5, nz=1.5, t=0.1)": {
            "nx": 4.5, "ny": 1.5, "nz": 1.5, "t": 0.1,
        },
        "Płytowy / Warstwowy (nx=1.5, ny=1.5, nz=4.5, t=-0.2)": {
            "nx": 1.5, "ny": 1.5, "nz": 4.5, "t": -0.2,
        },
        "Ultra-sztywny Octet-Truss (Schoen F-RD, kratownica)": {
            "tpms": "Octet", "tpms_label": "🏗️ Octet-Truss (Schoen F-RD)",
            "mode": "network", "mode_label": "🧱 Network (Lity szkielet f ≤ t)",
            "nx": 2.0, "ny": 2.0, "nz": 2.0, "t": 0.0,
        },
        "Wielokanałowy Neovius (Schoen C(P), 12 kanałów)": {
            "tpms": "Neovius", "tpms_label": "🕸️ Neovius (Schoen C(P))",
            "mode": "network", "mode_label": "🧱 Network (Lity szkielet f ≤ t)",
            "nx": 2.0, "ny": 2.0, "nz": 2.0, "t": 0.0,
        },
        "Asymetryczny labirynt I-WP (Schoen I-WP, 2 płyny)": {
            "tpms": "I-WP", "tpms_label": "🔀 Schoen I-WP (Asymetryczny)",
            "mode": "network", "mode_label": "🧱 Network (Lity szkielet f ≤ t)",
            "nx": 2.0, "ny": 2.0, "nz": 2.0, "t": 0.0,
        },
        "Pochłaniacz energii (Auksetyk, nu < 0)": {
            "tpms": "Auxetic", "tpms_label": "🎀 Re-entrant Auxetic (ν < 0)",
            "mode": "network", "mode_label": "🧱 Network (Lity szkielet f ≤ t)",
            "nx": 2.0, "ny": 2.0, "nz": 2.0, "t": 0.0,
        },
        "Wymiennik ciepła (Primitive Sheet, wysoki przepływ)": {
            "tpms": "Primitive", "tpms_label": "📦 Schwarz Primitive (P)",
            "mode": "sheet", "mode_label": "📜 Sheet (Ścianka powłokowa |f| ≤ tw)",
            "wall_thickness": 0.18,
            "nx": 2.0, "ny": 2.0, "nz": 2.0, "t": 0.0,
        },
        "Ultra-sztywny Hashin-Shtrikman (Diamond Sheet)": {
            "tpms": "Diamond", "tpms_label": "💎 Schwarz Diamond (D)",
            "mode": "sheet", "mode_label": "📜 Sheet (Ścianka powłokowa |f| ≤ tw)",
            "wall_thickness": 0.22,
            "nx": 2.0, "ny": 2.0, "nz": 2.0, "t": 0.0,
        },
        "Wysoka porowatość (t=-0.7)": {
            "nx": 2.5, "ny": 2.5, "nz": 2.5, "t": -0.7,
        },
        "Gęsta struktura ściankowa (t=0.6)": {
            "nx": 2.5, "ny": 2.5, "nz": 2.5, "t": 0.6,
        },
    }
    cfg = preset_configs.get(preset_name)
    if cfg:
        if "tpms" in cfg:
            st.session_state.selected_tpms = cfg["tpms"]
            st.session_state.sidebar_tpms_choice = cfg["tpms_label"]
        if "mode" in cfg:
            st.session_state.selected_mode = cfg["mode"]
            st.session_state.sidebar_mode_choice = cfg["mode_label"]
        if "wall_thickness" in cfg:
            st.session_state.wall_thickness = cfg["wall_thickness"]
            st.session_state.sidebar_wall_thickness = cfg["wall_thickness"]
        st.session_state.slider_nx = float(cfg["nx"])
        st.session_state.slider_ny = float(cfg["ny"])
        st.session_state.slider_nz = float(cfg["nz"])
        st.session_state.slider_t = float(cfg["t"])


def init_session_state():
    """Inicjalizacja modelu sieci i domyślnych danych w st.session_state."""
    if "selected_tpms" not in st.session_state:
        st.session_state.selected_tpms = "Gyroid"
    if "sidebar_tpms_choice" not in st.session_state:
        st.session_state.sidebar_tpms_choice = "🧊 Schoen Gyroid (G)"

    if "selected_mode" not in st.session_state:
        st.session_state.selected_mode = "network"
    if "sidebar_mode_choice" not in st.session_state:
        st.session_state.sidebar_mode_choice = "🧱 Network (Lity szkielet f ≤ t)"

    if "wall_thickness" not in st.session_state:
        st.session_state.wall_thickness = 0.20
    if "sidebar_wall_thickness" not in st.session_state:
        st.session_state.sidebar_wall_thickness = 0.20

    if "slider_nx" not in st.session_state:
        st.session_state.slider_nx = 2.0
    if "slider_ny" not in st.session_state:
        st.session_state.slider_ny = 2.0
    if "slider_nz" not in st.session_state:
        st.session_state.slider_nz = 2.0
    if "slider_t" not in st.session_state:
        st.session_state.slider_t = 0.0
    if "preset_selector" not in st.session_state:
        st.session_state.preset_selector = "Własna konfiguracja (Custom)"
    if "gallery_opacity" not in st.session_state:
        st.session_state.gallery_opacity = 0.85
    if "gallery_show_edges" not in st.session_state:
        st.session_state.gallery_show_edges = True
    if "gallery_show_bbox" not in st.session_state:
        st.session_state.gallery_show_bbox = True

    if "model" not in st.session_state:
        st.session_state.model = nm.StiffnessSurrogateModel(
            hidden_dims=[64, 128, 128, 64],
            lr=0.003,
            activation="gelu",
        )

    if "dataset_X" not in st.session_state or "dataset_y" not in st.session_state:
        X_init, y_init = tg.generate_synthetic_dataset(num_samples=500, random_seed=42, tpms_type="gyroid")
        st.session_state.dataset_X = X_init
        st.session_state.dataset_y = y_init

    if "auto_trained" not in st.session_state:
        checkpoint_path = "default_model.pth"
        if os.path.exists(checkpoint_path):
            try:
                st.session_state.model.load_checkpoint(checkpoint_path)
            except Exception:
                st.session_state.model.train_model(
                    st.session_state.dataset_X,
                    st.session_state.dataset_y,
                    epochs=70,
                    batch_size=64,
                )
                st.session_state.model.save_checkpoint(checkpoint_path)
        else:
            st.session_state.model.train_model(
                st.session_state.dataset_X,
                st.session_state.dataset_y,
                epochs=70,
                batch_size=64,
            )
            st.session_state.model.save_checkpoint(checkpoint_path)
        st.session_state.auto_trained = True


init_session_state()

# ==========================================
# Nagłówek aplikacji
# ==========================================
st.markdown(
    """
    <div class="main-header">
        <h1>🧊 💎 📦 🎀 🕸️ 🔀 🏗️ TPMS & Metamaterial Studio</h1>
        <p>Zaawansowane modelowanie i mikromechanika <b>7 kanonicznych struktur metamateriałowych</b>: 
        powierzchnie minimalne TPMS (<b>Schoen Gyroid</b>, <b>Schwarz Diamond</b>, <b>Schwarz Primitive</b>, <b>Neovius C(P)</b>, <b>Schoen I-WP</b>), 
        struktury kratowe (<b>Octet-Truss F-RD</b>) oraz auksetyczne (<b>Re-entrant Auxetic, ν < 0</b>) 
        w morfologii <b>Network (Bryła)</b> i <b>Sheet (Ścianka)</b> z aproksymacją tensora sztywności 6×6 w PyTorch.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# Pasek boczny (Sidebar) - Sterowanie
# ==========================================
with st.sidebar:
    st.header("⚙️ Konfiguracja Metamateriału")
    
    tpms_options = [
        "🧊 Schoen Gyroid (G)",
        "💎 Schwarz Diamond (D)",
        "📦 Schwarz Primitive (P)",
        "🎀 Re-entrant Auxetic (ν < 0)",
        "🕸️ Neovius (Schoen C(P))",
        "🔀 Schoen I-WP (Asymetryczny)",
        "🏗️ Octet-Truss (Schoen F-RD)",
    ]
    cur_tpms_choice = st.session_state.get("sidebar_tpms_choice", tpms_options[0])
    default_idx = tpms_options.index(cur_tpms_choice) if cur_tpms_choice in tpms_options else 0

    tpms_choice = st.radio(
        "Wybór geometrii metamateriału:",
        tpms_options,
        index=default_idx,
        key="sidebar_tpms_choice",
    )
    if "Diamond" in tpms_choice:
        current_tpms = "diamond"
        st.session_state.selected_tpms = "Diamond"
    elif "Primitive" in tpms_choice:
        current_tpms = "primitive"
        st.session_state.selected_tpms = "Primitive"
    elif "Auxetic" in tpms_choice:
        current_tpms = "auxetic"
        st.session_state.selected_tpms = "Auxetic"
    elif "Neovius" in tpms_choice:
        current_tpms = "neovius"
        st.session_state.selected_tpms = "Neovius"
    elif "I-WP" in tpms_choice:
        current_tpms = "iwp"
        st.session_state.selected_tpms = "I-WP"
    elif "Octet" in tpms_choice:
        current_tpms = "octet"
        st.session_state.selected_tpms = "Octet"
    else:
        current_tpms = "gyroid"
        st.session_state.selected_tpms = "Gyroid"

    st.markdown("---")
    st.header("🏗️ Morfologia Komórki")
    mode_options = [
        "🧱 Network (Lity szkielet f ≤ t)",
        "📜 Sheet (Ścianka powłokowa |f| ≤ tw)",
    ]
    cur_mode_choice = st.session_state.get("sidebar_mode_choice", mode_options[0])
    cur_mode_idx = mode_options.index(cur_mode_choice) if cur_mode_choice in mode_options else 0
    mode_choice = st.radio(
        "Tryb budowy komórki:",
        mode_options,
        index=cur_mode_idx,
        key="sidebar_mode_choice",
    )
    current_mode = "sheet" if "Sheet" in mode_choice else "network"
    st.session_state.selected_mode = current_mode

    if current_mode == "sheet":
        wall_thickness_val = st.slider(
            "Grubość ścianki tw (Wall thickness):",
            min_value=0.05,
            max_value=0.45,
            value=float(st.session_state.get("wall_thickness", 0.20)),
            step=0.01,
            key="sidebar_wall_thickness",
        )
        st.session_state.wall_thickness = wall_thickness_val
        st.caption("💡 *W trybie Sheet komórka ma podwójną ściankę o wyższej sztywności (granica Hashina-Shtrikmana).*")
    else:
        wall_thickness_val = float(st.session_state.get("wall_thickness", 0.20))

    st.markdown("---")
    st.header("📐 Wymiary i Periodyczność")

    # Presety parametrów
    preset_choice = st.selectbox(
        "Wybierz predefiniowaną konfigurację (Preset):",
        [
            "Własna konfiguracja (Custom)",
            "Izotropowy standardowy (nx=2, ny=2, nz=2, t=0)",
            "Anizotropowy belkowy (nx=4.5, ny=1.5, nz=1.5, t=0.1)",
            "Płytowy / Warstwowy (nx=1.5, ny=1.5, nz=4.5, t=-0.2)",
            "Ultra-sztywny Octet-Truss (Schoen F-RD, kratownica)",
            "Wielokanałowy Neovius (Schoen C(P), 12 kanałów)",
            "Asymetryczny labirynt I-WP (Schoen I-WP, 2 płyny)",
            "Pochłaniacz energii (Auksetyk, nu < 0)",
            "Wymiennik ciepła (Primitive Sheet, wysoki przepływ)",
            "Ultra-sztywny Hashin-Shtrikman (Diamond Sheet)",
            "Wysoka porowatość (t=-0.7)",
            "Gęsta struktura ściankowa (t=0.6)",
        ],
        key="preset_selector",
        on_change=apply_preset,
    )

    nx = st.slider("Liczba komórek nx (Oś X):", min_value=1.0, max_value=5.0, value=float(st.session_state.slider_nx), step=0.1, key="slider_nx")
    ny = st.slider("Liczba komórek ny (Oś Y):", min_value=1.0, max_value=5.0, value=float(st.session_state.slider_ny), step=0.1, key="slider_ny")
    nz = st.slider("Liczba komórek nz (Oś Z):", min_value=1.0, max_value=5.0, value=float(st.session_state.slider_nz), step=0.1, key="slider_nz")
    t_val = st.slider("Parametr izopowierzchni t (Próg gęstości):", min_value=-1.0, max_value=1.0, value=float(st.session_state.slider_t), step=0.05, key="slider_t")

    st.markdown("---")
    st.header("🧠 Uczenie Sieci Neuronowej")
    num_samples = st.slider("Liczba próbek syntetycznych:", min_value=100, max_value=1500, value=600, step=50)
    train_epochs = st.slider("Liczba epok treningu:", min_value=20, max_value=250, value=100, step=10)
    learning_rate = st.select_slider(
        "Współczynnik uczenia (Learning Rate):",
        options=[0.0005, 0.001, 0.002, 0.005, 0.01],
        value=0.002,
    )
    noise_param = st.slider("Szum losowy danych (%):", min_value=0.0, max_value=5.0, value=1.5, step=0.5) / 100.0

    train_btn = st.button("🚀 Trenuj Sieć dla Aktywnej Konfiguracji", width="stretch", type="primary")

    st.markdown("---")
    mesh_res = st.select_slider("Rozdzielczość siatki 3D Isosurface:", options=[24, 30, 36, 42], value=30)
    
    active_label_map = {
        "gyroid": "Gyroid (Schoen G)",
        "diamond": "Diamond (Schwarz D)",
        "primitive": "Primitive (Schwarz P)",
        "auxetic": "Auxetic (Re-entrant, ν < 0)",
        "neovius": "Neovius (Schoen C(P))",
        "iwp": "Schoen I-WP (Asymetryczny)",
        "octet": "Octet-Truss (Schoen F-RD)",
    }
    st.info(
        f"**Aktywna struktura:** `{active_label_map[current_tpms]}`\n\n"
        f"**Tryb morfologii:** `{'📜 Sheet' if current_mode == 'sheet' else '🧱 Network'}`\n\n"
        f"**Status modelu:** {'✅ Wytrenowany' if st.session_state.model.is_trained else '⏳ Oczekuje na trening'}\n\n"
        f"**Urządzenie PyTorch:** `{st.session_state.model.device}`"
    )

# Obsługa przycisku treningu
if train_btn:
    with st.spinner(f"Generowanie danych syntetycznych ({active_label_map[current_tpms]}, {current_mode.upper()}) i trening sieci MLP..."):
        X_new, y_new = tg.generate_synthetic_dataset(
            num_samples=num_samples,
            random_seed=int(time.time()),
            noise_level=noise_param,
            tpms_type=current_tpms,
            mode=current_mode,
            wall_thickness=wall_thickness_val,
        )
        st.session_state.dataset_X = X_new
        st.session_state.dataset_y = y_new

        progress_bar = st.progress(0, text="Rozpoczynanie treningu...")

        def on_epoch_progress(epoch, total_epochs, train_loss, val_loss, val_r2):
            pct = int((epoch / total_epochs) * 100)
            progress_bar.progress(
                pct,
                text=f"Epoka {epoch}/{total_epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | R²: {val_r2:.4f}",
            )

        st.session_state.model.train_model(
            X_new,
            y_new,
            epochs=train_epochs,
            batch_size=32,
            learning_rate=learning_rate,
            progress_callback=on_epoch_progress,
        )
        progress_bar.empty()
        st.success(f"Ukończono trening! Ostateczny błąd walidacji: {st.session_state.model.history['val_loss'][-1]:.5f}")

# ==========================================
# Obliczenia mechaniczne i predykcja NN
# ==========================================
current_input = np.array([nx, ny, nz, t_val], dtype=np.float32)

# Macierz wzorcowa (Ground Truth z syntetycznego homogenizatora)
C_true = tg.synthetic_stiffness_matrix(
    nx, ny, nz, t_val,
    noise_level=0.0,
    tpms_type=current_tpms,
    mode=current_mode,
    wall_thickness=wall_thickness_val,
)

# Predykcja sieci neuronowej
if st.session_state.model.is_trained:
    _, C_pred = st.session_state.model.predict(current_input)
else:
    C_pred = np.zeros_like(C_true)

# Macierz błędu bezwzględnego
C_diff = np.abs(C_true - C_pred)
mae_error = float(np.mean(C_diff))
max_error = float(np.max(C_diff))
rel_error_pct = float(np.mean(C_diff / (np.abs(C_true) + 1e-6)) * 100.0)

# Gęstość względna
rho_calc = tg.compute_relative_density(
    nx, ny, nz, t_val,
    resolution=25,
    tpms_type=current_tpms,
    mode=current_mode,
    wall_thickness=wall_thickness_val,
)

# ==========================================
# Pasek kluczowych metryk (KPIs)
# ==========================================
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Gęstość względna (ρ)</div>
            <div class="metric-value">{rho_calc * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">C11 (Oś X) True / Pred</div>
            <div class="metric-value">{C_true[0, 0]:.1f} / {C_pred[0, 0]:.1f} <span style="font-size:0.9rem; color:#94a3b8;">GPa</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">C22 (Oś Y) True / Pred</div>
            <div class="metric-value">{C_true[1, 1]:.1f} / {C_pred[1, 1]:.1f} <span style="font-size:0.9rem; color:#94a3b8;">GPa</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">C33 (Oś Z) True / Pred</div>
            <div class="metric-value">{C_true[2, 2]:.1f} / {C_pred[2, 2]:.1f} <span style="font-size:0.9rem; color:#94a3b8;">GPa</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m5:
    if current_tpms == "auxetic":
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">C12 (Poprzeczne ν < 0)</div>
                <div class="metric-value">{C_true[0, 1]:.1f} <span style="font-size:0.9rem; color:#38bdf8;">GPa (Auksetyk!)</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Średni Błąd Sieci (MAE)</div>
                <div class="metric-value">{mae_error:.2f} <span style="font-size:0.9rem; color:#f87171;">GPa ({rel_error_pct:.1f}%)</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==========================================
# Zakładki Główne Dashboardu
# ==========================================
tab_3d_geom, tab_matrix, tab_young, tab_training, tab_dataset = st.tabs(
    [
        "🧊/💎/📦/🎀/🕸️/🔀/🏗️ 1. Morfologia & Modele 3D (7 Architektur)",
        "📊 2. Macierze Sztywności 6×6 (Porównanie)",
        "🌐 3. Kierunkowy Moduł Younga 3D (Anizotropia)",
        "📈 4. Diagnostyka Uczenia Sieci",
        "💾 5. Zbiór Danych & Eksport 3D (STEP/IGES/STL)",
    ]
)

# ----------------------------------------------------
# TAB 1: Morfologia i Modele 3D: 7 Kanonicznych Architektur
# ----------------------------------------------------
with tab_3d_geom:
    st.subheader(f"Trójwymiarowy Model Metamateriału: 7 Kanonicznych Struktur ({'Ściankowy (Sheet)' if current_mode == 'sheet' else 'Szkieletowy (Network)'})")

    col_view_opt, col_color_opt = st.columns([2, 1])
    with col_view_opt:
        tpms_view_mode = st.radio(
            "Wybierz widok geometrii:",
            [
                f"Aktywna struktura: {active_label_map[current_tpms]}",
                "🧊 Schoen Gyroid (G)",
                "💎 Schwarz Diamond (D)",
                "📦 Schwarz Primitive (P)",
                "🎀 Re-entrant Auxetic (ν < 0)",
                "🕸️ Neovius (Schoen C(P))",
                "🔀 Schoen I-WP (Asymetryczny)",
                "🏗️ Octet-Truss (Schoen F-RD)",
                "⚖️ Porównanie Wszystkich 7 Struktur (Galeria)",
            ],
            index=0,
            horizontal=True,
            key="tab1_view_mode",
        )
    with col_color_opt:
        color_scheme = st.selectbox(
            "Paleta barw Isosurface:",
            ["Viridis", "Plasma", "Cividis", "Turbo", "Blues", "Magma"],
            index=0,
            key="iso_color_scheme",
        )

    # Obliczenie gęstości dla wszystkich 7 struktur
    rho_gyroid = tg.compute_relative_density(nx, ny, nz, t_val, resolution=22, tpms_type="gyroid", mode=current_mode, wall_thickness=wall_thickness_val)
    rho_diamond = tg.compute_relative_density(nx, ny, nz, t_val, resolution=22, tpms_type="diamond", mode=current_mode, wall_thickness=wall_thickness_val)
    rho_primitive = tg.compute_relative_density(nx, ny, nz, t_val, resolution=22, tpms_type="primitive", mode=current_mode, wall_thickness=wall_thickness_val)
    rho_auxetic = tg.compute_relative_density(nx, ny, nz, t_val, resolution=22, tpms_type="auxetic", mode=current_mode, wall_thickness=wall_thickness_val)
    rho_neovius = tg.compute_relative_density(nx, ny, nz, t_val, resolution=22, tpms_type="neovius", mode=current_mode, wall_thickness=wall_thickness_val)
    rho_iwp = tg.compute_relative_density(nx, ny, nz, t_val, resolution=22, tpms_type="iwp", mode=current_mode, wall_thickness=wall_thickness_val)
    rho_octet = tg.compute_relative_density(nx, ny, nz, t_val, resolution=22, tpms_type="octet", mode=current_mode, wall_thickness=wall_thickness_val)

    # 1. TRYB PORÓWNANIA OBOK SIEBIE (WSZYSTKIE 7 STRUKTUR)
    if tpms_view_mode == "⚖️ Porównanie Wszystkich 7 Struktur (Galeria)":
        st.markdown(
            f"""
            #### ⚖️ Pełne Zestawienie Topologii 7 Struktur Metamateriałowych (Tryb: **{current_mode.upper()}**)
            Zestawienie właściwości topologicznych, symetrii krystalograficznych oraz zachowania mikromechanicznego wygenerowanych dla $[n_x, n_y, n_z] = [{nx:.1f}, {ny:.1f}, {nz:.1f}]$, $t={t_val:.2f}$ oraz $t_w={wall_thickness_val:.2f}$.
            """
        )

        # Kontrolki wizualizacji dla widoku galerii (przezroczystość i krawędzie)
        st.markdown("##### 🎛️ Opcje renderowania galerii (Przezroczystość i Krawędzie):")
        col_gal_opt1, col_gal_opt2, col_gal_opt3 = st.columns([2, 1, 1])
        with col_gal_opt1:
            gallery_opacity = st.slider(
                "Przezroczystość powierzchni (Opacity):",
                min_value=0.10,
                max_value=1.00,
                value=float(st.session_state.get("gallery_opacity", 0.85)),
                step=0.05,
                key="gallery_opacity_slider",
                help="Regulacja przezroczystości izopowierzchni we wszystkich 7 modelach w galerii.",
            )
            st.session_state.gallery_opacity = gallery_opacity
        with col_gal_opt2:
            gallery_show_edges = st.checkbox(
                "Włącz krawędzie (Contours / Edges)",
                value=bool(st.session_state.get("gallery_show_edges", True)),
                key="gallery_show_edges_checkbox",
                help="Włącza wyświetlanie linii konturowych i krawędzi na powierzchni modeli 3D.",
            )
            st.session_state.gallery_show_edges = gallery_show_edges
        with col_gal_opt3:
            gallery_show_bbox = st.checkbox(
                "Krawędzie komórki (Bounding Box)",
                value=bool(st.session_state.get("gallery_show_bbox", True)),
                key="gallery_show_bbox_checkbox",
                help="Wyświetla krawędzie sześcianu komórki elementarnej [0, 1]³.",
            )
            st.session_state.gallery_show_bbox = gallery_show_bbox

        st.markdown(
            f"""
            | Parametr / Cecha | 🧊 Schoen Gyroid (G) | 💎 Schwarz Diamond (D) | 📦 Schwarz Primitive (P) | 🎀 Re-entrant Auxetic (ν < 0) | 🕸️ Neovius (Schoen C(P)) | 🔀 Schoen I-WP | 🏗️ Octet-Truss (Schoen F-RD) |
            |---|---|---|---|---|---|---|---|
            | **Grupa krystalograficzna** | $Ia\\bar{{3}}d$ (nr 230) | $Pn\\bar{{3}}m$ (nr 224) | $Im\\bar{{3}}m$ (nr 229) | Re-entrant ($m\\bar{{3}}m$) | $Im\\bar{{3}}m$ (nr 229) | $Im\\bar{{3}}m$ (nr 229) | $Fm\\bar{{3}}m$ (nr 225) |
            | **Współczynnik Poissona (ν)** | $\\sim +0.28$ | $\\sim +0.30$ | $\\sim +0.25$ | **$\\sim -0.22$ (UJEMNY)** | $\\sim +0.27$ | $\\sim +0.29$ | $\\sim +0.33$ |
            | **Topologia sieci / kanałów** | 2 splecione labirynty chiralne | Tetraedryczne węzły $sp^3$ | 6 otwartych kanałów osiowych | Węzły wklęsłe (re-entrant) | 12 kanałów diagonalnych | 2 asymetryczne labirynty (I i WP) | Krystaliczna krata oktetowa FCC |
            | **Gęstość względna (ρ)** | **{rho_gyroid * 100:.1f}%** | **{rho_diamond * 100:.1f}%** | **{rho_primitive * 100:.1f}%** | **{rho_auxetic * 100:.1f}%** | **{rho_neovius * 100:.1f}%** | **{rho_iwp * 100:.1f}%** | **{rho_octet * 100:.1f}%** |
            | **Mechanizm odkształceń** | Zginanie ścianek ($m \\approx 1.85$) | Ściskanie węzłów ($m \\approx 1.76$) | Ortotropia osiowa ($m \\approx 1.95$) | Rozszerzanie poprzeczne | Ścinanie diagonalne ($m \\approx 1.80$) | Quasi-izotropowy ($m \\approx 1.74$) | **Rozciąganie prętów ($m \\approx 1.55$)** |
            | **Zastosowania inżynierskie** | Wymienniki ciepła, implanty | Konstrukcje nośne, lotnictwo | Filtry cieczowe, katalizatory | Ochrona balistyczna, absorpcja | Radiatory wysokopowierzchniowe | Dwufazowe wymienniki, separatory | Ramy ultralekkie o maks. sztywności |
            """
        )

        # Rząd 1: Gyroid i Diamond
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"##### 🧊 Schoen Gyroid ($Ia\\bar{{3}}d$) — ρ ≈ {rho_gyroid*100:.1f}%")
            X_g, Y_g, Z_g, F_g = get_cached_gyroid_grid(nx, ny, nz, t_val, resolution=mesh_res, mode=current_mode, wall_thickness=wall_thickness_val)
            fig_g = build_gallery_figure(X_g, Y_g, Z_g, F_g, "Blues", opacity=gallery_opacity, show_edges=gallery_show_edges, show_bbox=gallery_show_bbox)
            st.plotly_chart(fig_g, width="stretch")

        with col_c2:
            st.markdown(f"##### 💎 Schwarz Diamond ($Pn\\bar{{3}}m$) — ρ ≈ {rho_diamond*100:.1f}%")
            X_d, Y_d, Z_d, F_d = get_cached_diamond_grid(nx, ny, nz, t_val, resolution=mesh_res, mode=current_mode, wall_thickness=wall_thickness_val)
            fig_d = build_gallery_figure(X_d, Y_d, Z_d, F_d, "Plasma", opacity=gallery_opacity, show_edges=gallery_show_edges, show_bbox=gallery_show_bbox)
            st.plotly_chart(fig_d, width="stretch")

        # Rząd 2: Primitive i Auxetic
        col_c3, col_c4 = st.columns(2)
        with col_c3:
            st.markdown(f"##### 📦 Schwarz Primitive ($Im\\bar{{3}}m$) — ρ ≈ {rho_primitive*100:.1f}%")
            X_p, Y_p, Z_p, F_p = get_cached_primitive_grid(nx, ny, nz, t_val, resolution=mesh_res, mode=current_mode, wall_thickness=wall_thickness_val)
            fig_p = build_gallery_figure(X_p, Y_p, Z_p, F_p, "Cividis", opacity=gallery_opacity, show_edges=gallery_show_edges, show_bbox=gallery_show_bbox)
            st.plotly_chart(fig_p, width="stretch")

        with col_c4:
            st.markdown(f"##### 🎀 Re-entrant Auxetic (ν < 0) — ρ ≈ {rho_auxetic*100:.1f}%")
            X_a, Y_a, Z_a, F_a = get_cached_auxetic_grid(nx, ny, nz, t_val, resolution=mesh_res, mode=current_mode, wall_thickness=wall_thickness_val)
            fig_a = build_gallery_figure(X_a, Y_a, Z_a, F_a, "Viridis", opacity=gallery_opacity, show_edges=gallery_show_edges, show_bbox=gallery_show_bbox)
            st.plotly_chart(fig_a, width="stretch")

        # Rząd 3: Neovius i Schoen I-WP
        col_c5, col_c6 = st.columns(2)
        with col_c5:
            st.markdown(f"##### 🕸️ Neovius Schoen C(P) ($Im\\bar{{3}}m$) — ρ ≈ {rho_neovius*100:.1f}%")
            X_n, Y_n, Z_n, F_n = get_cached_neovius_grid(nx, ny, nz, t_val, resolution=mesh_res, mode=current_mode, wall_thickness=wall_thickness_val)
            fig_n = build_gallery_figure(X_n, Y_n, Z_n, F_n, "Turbo", opacity=gallery_opacity, show_edges=gallery_show_edges, show_bbox=gallery_show_bbox)
            st.plotly_chart(fig_n, width="stretch")

        with col_c6:
            st.markdown(f"##### 🔀 Schoen I-WP ($Im\\bar{{3}}m$) — ρ ≈ {rho_iwp*100:.1f}%")
            X_i, Y_i, Z_i, F_i = get_cached_iwp_grid(nx, ny, nz, t_val, resolution=mesh_res, mode=current_mode, wall_thickness=wall_thickness_val)
            fig_i = build_gallery_figure(X_i, Y_i, Z_i, F_i, "Magma", opacity=gallery_opacity, show_edges=gallery_show_edges, show_bbox=gallery_show_bbox)
            st.plotly_chart(fig_i, width="stretch")

        # Rząd 4: Octet-Truss (Schoen F-RD)
        col_c7, col_c8 = st.columns([1, 1])
        with col_c7:
            st.markdown(f"##### 🏗️ Octet-Truss Schoen F-RD ($Fm\\bar{{3}}m$) — ρ ≈ {rho_octet*100:.1f}%")
            X_o, Y_o, Z_o, F_o = get_cached_octet_grid(nx, ny, nz, t_val, resolution=mesh_res, mode=current_mode, wall_thickness=wall_thickness_val)
            fig_o = build_gallery_figure(X_o, Y_o, Z_o, F_o, "Plasma", opacity=gallery_opacity, show_edges=gallery_show_edges, show_bbox=gallery_show_bbox)
            st.plotly_chart(fig_o, width="stretch")
        with col_c8:
            st.markdown(
                r"""
                <div class="formula-box" style="margin-top: 2rem;">
                    <h4>🚀 Porównanie Efektywności Strukturalnej</h4>
                    <p><b>1. Mechanizm zginania (Bending-dominated):</b> Schoen Gyroid, Primitive, Neovius wykazują skalowanie sztywności $E \propto \rho^m$ z wykładnikiem $m \approx 1.8-2.0$. Idealne do absorpcji energii i łagodnego zgniatania.</p>
                    <p><b>2. Mechanizm rozciągania (Stretching-dominated):</b> Octet-Truss charakteryzuje się niemal liniowym skalowaniem $E \propto \rho^{1.1-1.5}$. Przy danej masie oferuje najwyższą sztywność konstrukcyjną w osiach głównych.</p>
                    <p><b>3. Wpływ morfologii Sheet vs Network:</b> Wersje powłokowe (Sheet) przesuwają mechanikę w stronę granicy Hashina-Shtrikmana, podwajając sztywność w stosunku do Network o tej samej gęstości.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 2. TRYB POJEDYNCZEJ STRUKTURY
    else:
        if "Diamond" in tpms_view_mode:
            chosen_single = "diamond"
        elif "Primitive" in tpms_view_mode:
            chosen_single = "primitive"
        elif "Auxetic" in tpms_view_mode:
            chosen_single = "auxetic"
        elif "Neovius" in tpms_view_mode:
            chosen_single = "neovius"
        elif "I-WP" in tpms_view_mode:
            chosen_single = "iwp"
        elif "Octet" in tpms_view_mode:
            chosen_single = "octet"
        elif "Gyroid" in tpms_view_mode:
            chosen_single = "gyroid"
        else:
            chosen_single = current_tpms

        if chosen_single == "diamond":
            st.markdown(
                """
                **Struktura: 💎 Schwarz Diamond (D-surface)**  
                $$f_D(x, y, z) = \\sin(k_x x)\\sin(k_y y)\\sin(k_z z) + \\sin(k_x x)\\cos(k_y y)\\cos(k_z z) + \\cos(k_x x)\\sin(k_y y)\\cos(k_z z) + \\cos(k_x x)\\cos(k_y y)\\sin(k_z z) - t = 0$$
                Należy do grupy krystalograficznej $Pn\\bar{3}m$. Charakteryzuje się 4-ramiennymi węzłami tetraedrycznymi o bardzo wysokiej sztywności na ściskanie osiowe i ścinanie.
                """
            )
            active_rho = rho_diamond
        elif chosen_single == "primitive":
            st.markdown(
                """
                **Struktura: 📦 Schwarz Primitive (P-surface)**  
                $$f_P(x, y, z) = \\cos(k_x x) + \\cos(k_y y) + \\cos(k_z z) - t = 0$$
                Dopełnia tzw. *Wielką Trójkę* TPMS. Należy do grupy $Im\\bar{3}m$. Tworzy prostą, otwartą sieć kanałów prostopadłych o najniższych oporach przepływu, idealną do wymienników ciepła i membran.
                """
            )
            active_rho = rho_primitive
        elif chosen_single == "auxetic":
            st.markdown(
                """
                **Struktura: 🎀 Re-entrant Auxetic (Auksetyk 3D, ν < 0)**  
                $$f_{aux}(x, y, z) = \\cos(k_x x)\\cos(k_y y)\\cos(k_z z) - c\\cdot(\\cos(2k_x x) + \\cos(2k_y y) + \\cos(2k_z z)) - t = 0$$
                Metamateriał o **ujemnym współczynniku Poissona ($\nu < 0$)**. Wklęsłe krawędzie komórki rozprostowują się na zewnątrz przy rozciąganiu, dając niespotykane pochłanianie energii i odporność na pękanie.
                """
            )
            active_rho = rho_auxetic
        elif chosen_single == "neovius":
            st.markdown(
                r"""
                **Struktura: 🕸️ Neovius (Schoen C(P) surface)**  
                $$f_N(x, y, z) = \frac{1}{3}\left[3(\cos(k_x x) + \cos(k_y y) + \cos(k_z z)) + 4\cos(k_x x)\cos(k_y y)\cos(k_z z)\right] - t = 0$$
                Powierzchnia minimalna o grupie symetrii $Im\bar{3}m$. Posiada 12 kanałów diagonalnych przecinających węzły sześcianu, oferując znakomitą sztywność na ścinanie ($\gamma_{shear} \approx 0.44$) oraz bardzo dużą powierzchnię wymiany masy i ciepła.
                """
            )
            active_rho = rho_neovius
        elif chosen_single == "iwp":
            st.markdown(
                """
                **Struktura: 🔀 Schoen I-WP (I-Wrapped Package)**  
                $$f_{IWP}(x, y, z) = 2(\\cos(k_x x)\\cos(k_y y) + \\cos(k_y y)\\cos(k_z z) + \\cos(k_z z)\\cos(k_x x)) - (\\cos(2k_x x) + \\cos(2k_y y) + \\cos(2k_z z)) - t = 0$$
                Powierzchnia minimalna o grupie $Im\\bar{3}m$ dzieląca przestrzeń na **dwa asymetryczne labirynty** (sieć I oraz sieć WP). Charakteryzuje się wysoką quasi-izotropią mechaniczną ($\alpha \\approx 1.12$) i znakomitą wydajnością w wymiennikach dwufazowych.
                """
            )
            active_rho = rho_iwp
        elif chosen_single == "octet":
            st.markdown(
                """
                **Struktura: 🏗️ Octet-Truss (Schoen F-RD / Krata Oktetowa)**  
                $$f_{octet}(x, y, z) = 4\\cos(k_x x)\\cos(k_y y)\\cos(k_z z) - (\\cos(2k_x x)\\cos(2k_y y) + \\cos(2k_y y)\\cos(2k_z z) + \\cos(2k_z z)\\cos(2k_x x)) - t = 0$$
                Gładka powierzchnia Schoen F-RD odzwierciedlająca **kratownicę oktetową FCC** (Deshpande-Fleck). Posiada mechanizm odkształceń zdominowany przez rozciąganie prętów (**stretching-dominated**, $m \\approx 1.55$ w Network, $m \\approx 1.10$ w Sheet), oferując najwyższy stosunek modułu Younga do gęstości.
                """
            )
            active_rho = rho_octet
        else:
            st.markdown(
                """
                **Struktura: 🧊 Schoen Gyroid (G-surface)**  
                $$f_G(x, y, z) = \\sin(k_x x)\\cos(k_y y) + \\sin(k_y y)\\cos(k_z z) + \\sin(k_z z)\\cos(k_x x) - t = 0$$
                Powierzchnia należy do grupy $Ia\\bar{3}d$. Jest strukturą chiralną o zerowej średniej krzywiźnie ($H=0$), tworzącą dwa splecione bezkontaktowe labirynty o doskonałej ciągłości naprężeń.
                """
            )
            active_rho = rho_gyroid

        col_iso_opt1, col_iso_opt2 = st.columns([1, 3])
        with col_iso_opt1:
            opacity_val = st.slider("Przezroczystość powierzchni:", 0.10, 1.00, 0.95, 0.05, key="iso_opacity_slider")
            show_caps = st.checkbox("Zamknij ściany komórki (Caps)", value=False, key="iso_caps_check")
            show_edges_single = st.checkbox("Włącz krawędzie (Contours / Edges)", value=False, key="iso_edges_check", help="Włącza linie konturowe i krawędzie powierzchni.")
            show_bbox_single = st.checkbox("Ramka komórki (Bounding Box)", value=False, key="iso_bbox_check", help="Wyświetla krawędzie sześcianu komórki elementarnej [0, 1]³.")

            st.markdown(
                f"""
                <div class="formula-box">
                    <b>Parametry komórki:</b><br>
                    • Struktura: <b>{active_label_map[chosen_single]}</b><br>
                    • Morfologia: <b>{'📜 Sheet (Ścianka)' if current_mode == 'sheet' else '🧱 Network (Bryła)'}</b><br>
                    • Periodyczność: nx=<b>{nx:.2f}</b>, ny=<b>{ny:.2f}</b>, nz=<b>{nz:.2f}</b><br>
                    • Wartość progu t = <b>{t_val:.2f}</b><br>
                    • Gęstość względna ρ ≈ <b>{active_rho:.3f} ({active_rho*100:.1f}%)</b><br>
                    • Rozdzielczość 3D: <b>{mesh_res}³ = {mesh_res**3:,}</b> punktów
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Szybki przełącznik aktywnej struktury w dashboardzie
            if chosen_single != current_tpms:
                label_target_map = {
                    "diamond": ("Diamond", "💎 Schwarz Diamond (D)"),
                    "primitive": ("Primitive", "📦 Schwarz Primitive (P)"),
                    "auxetic": ("Auxetic", "🎀 Re-entrant Auxetic (ν < 0)"),
                    "neovius": ("Neovius", "🕸️ Neovius (Schoen C(P))"),
                    "iwp": ("I-WP", "🔀 Schoen I-WP (Asymetryczny)"),
                    "octet": ("Octet", "🏗️ Octet-Truss (Schoen F-RD)"),
                    "gyroid": ("Gyroid", "🧊 Schoen Gyroid (G)"),
                }
                target_label, target_radio = label_target_map.get(chosen_single, ("Gyroid", "🧊 Schoen Gyroid (G)"))
                if st.button(f"Ustaw {target_label} jako aktywny model obliczeniowy", type="primary", width="stretch", key=f"btn_set_active_{chosen_single}"):
                    st.session_state.selected_tpms = target_label
                    st.session_state.sidebar_tpms_choice = target_radio
                    st.session_state.preset_selector = "Własna konfiguracja (Custom)"
                    st.rerun()

        with col_iso_opt2:
            X_grid, Y_grid, Z_grid, F_grid = get_cached_tpms_grid(
                chosen_single, nx, ny, nz, t_val,
                resolution=mesh_res,
                mode=current_mode,
                wall_thickness=wall_thickness_val,
            )

            contour_cfg_single = dict(show=True, color="#0f172a", width=3) if show_edges_single else dict(show=False)
            fig_iso = go.Figure(
                data=go.Isosurface(
                    x=X_grid.flatten(),
                    y=Y_grid.flatten(),
                    z=Z_grid.flatten(),
                    value=F_grid.flatten(),
                    isomin=-0.08,
                    isomax=0.08,
                    surface_count=2,
                    colorscale=color_scheme,
                    opacity=opacity_val,
                    caps=dict(x_show=show_caps, y_show=show_caps, z_show=show_caps),
                    contour=contour_cfg_single,
                    colorbar=dict(title="f(x,y,z)", thickness=15, len=0.7),
                )
            )
            if show_bbox_single:
                fig_iso.add_trace(get_bounding_box_trace(Lx=1.0, Ly=1.0, Lz=1.0, color="#475569", width=2.5))

            fig_iso.update_layout(
                scene=dict(
                    xaxis=dict(title="X [Lx]", nticks=5),
                    yaxis=dict(title="Y [Ly]", nticks=5),
                    zaxis=dict(title="Z [Lz]", nticks=5),
                    camera=dict(eye=dict(x=1.6, y=1.6, z=1.4)),
                    aspectmode="cube",
                ),
                margin=dict(l=0, r=0, b=0, t=10),
                height=600,
            )
            st.plotly_chart(fig_iso, width="stretch")

    # Sekcja Szybkiego Eksportu 3D
    with st.expander("📦 Szybki Eksport Aktywnego Modelu 3D (STEP / IGES / STL / OBJ)", expanded=False):
        chosen_export_tpms = chosen_single if ('chosen_single' in locals() and not tpms_view_mode.startswith("⚖️")) else current_tpms
        st.markdown(
            f"Pobierz wygenerowaną geometrię **{active_label_map[chosen_export_tpms]}** w trybie **{current_mode.upper()}** "
            f"w formatach CAD (STEP, IGES) oraz siatkach 3D (STL, OBJ) dla parametrów $n=[{nx:.1f}, {ny:.1f}, {nz:.1f}], t={t_val:.2f}$."
        )
        col_qe1, col_qe2, col_qe3, col_qe4 = st.columns(4)

        with st.spinner("Przygotowywanie siatki eksportowej..."):
            quick_mesh = exp3d.generate_tpms_mesh(
                tpms_type=chosen_export_tpms,
                nx=nx,
                ny=ny,
                nz=nz,
                t=t_val,
                resolution=18,
                size_x_mm=25.0,
                size_y_mm=25.0,
                size_z_mm=25.0,
                mode=current_mode,
                wall_thickness=wall_thickness_val,
            )

        with col_qe1:
            stl_bytes = exp3d.export_to_stl(quick_mesh, binary=True)
            st.download_button(
                label="📥 Pobierz STL (Druk 3D)",
                data=stl_bytes,
                file_name=f"{chosen_export_tpms}_{current_mode}_model.stl",
                mime="application/sla",
                width="stretch",
            )
        with col_qe2:
            if exp3d.OCP_AVAILABLE:
                step_bytes = exp3d.export_to_step(quick_mesh)
                st.download_button(
                    label="📥 Pobierz STEP (CAD ISO)",
                    data=step_bytes,
                    file_name=f"{chosen_export_tpms}_{current_mode}_model.step",
                    mime="application/step",
                    width="stretch",
                )
            else:
                st.info("Brak OCP dla STEP")
        with col_qe3:
            if exp3d.OCP_AVAILABLE:
                iges_bytes = exp3d.export_to_iges(quick_mesh)
                st.download_button(
                    label="📥 Pobierz IGES (CAD ANSI)",
                    data=iges_bytes,
                    file_name=f"{chosen_export_tpms}_{current_mode}_model.iges",
                    mime="application/iges",
                    width="stretch",
                )
            else:
                st.info("Brak OCP dla IGES")
        with col_qe4:
            obj_bytes = exp3d.export_to_obj(quick_mesh)
            st.download_button(
                label="📥 Pobierz OBJ (Siatka 3D)",
                data=obj_bytes,
                file_name=f"{chosen_export_tpms}_{current_mode}_model.obj",
                mime="text/plain",
                width="stretch",
            )

# ----------------------------------------------------
# TAB 2: Macierze Sztywności 6x6 (Heatmaps)
# ----------------------------------------------------
with tab_matrix:
    st.subheader(f"Zestawienie Macierzy Sztywności 6×6 w Notacji Voigta ({active_label_map[current_tpms]}, {current_mode.upper()}) [w GPa]")
    st.markdown(
        f"Porównanie rzeczywistej ortotropowej macierzy sztywności ($C_{{true}}$) dla struktury **{active_label_map[current_tpms]}** w trybie **{current_mode.upper()}**, predykcji sieci neuronowej ($C_{{pred}}$) oraz bezwzględnej macierzy różnic $|C_{{true}} - C_{{pred}}|$."
    )
    if current_tpms == "auxetic":
        st.info("🎀 **Zjawisko Auksetyczne (Ujemny Współczynnik Poissona ν < 0):** W macierzy sztywności elementy sprzężenia poprzecznego $C_{12}, C_{13}, C_{23} < 0$, co odpowiada dodatnim składowym macierzy podatności $S_{12} > 0$. Materiał przy rozciąganiu w jednym kierunku rozszerza się poprzecznie (brak przewężenia).")
    elif current_tpms == "octet":
        st.info("🏗️ **Kratownica Oktetowa (Stretching-Dominated FCC):** Odkształcenia osiowe prętów sprawiają, że moduł Younga $E$ i sztywności diagonalne $C_{11}, C_{22}, C_{33}$ skalują się niemal liniowo z gęstością względną ($m \\approx 1.55$ w Network, $m \\approx 1.10$ w Sheet), dając najwyższą sprawność sztywnościową.")
    elif current_tpms == "neovius":
        st.info("🕸️ **Neovius Schoen C(P) (Podwyższone Ścinanie):** 12 kanałów diagonalnych w komórce elementarnej przenosi znaczące składowe ścinające ($C_{44}, C_{55}, C_{66}$), dając ponadprzeciętną sztywność postaciową w porównaniu z klasycznymi strukturami TPMS.")
    elif current_tpms == "iwp":
        st.info("🔀 **Schoen I-WP (Podwójny Labirynt Asymetryczny):** Zbalansowana odpowiedź sprężysta w przestrzeni 3D z niskim wskaźnikiem anizotropii Zenera ($Z \\approx 1.12$), gwarantująca stabilną nośność wieloosiową.")
    if current_mode == "sheet":
        st.success("📜 **Tryb Powłokowy (Sheet TPMS — Granica Hashina-Shtrikmana):** Ścianki komórki pracują w stanie membranowym (stretching-dominated, $m \\approx 1.20$), zapewniając prawie 2-krotnie wyższy moduł Younga niż komórki bryłowe (Network) o zbliżonej masie.")

    labels = ["ε₁₁ (xx)", "ε₂₂ (yy)", "ε₃₃ (zz)", "γ₂₃ (yz)", "γ₁₃ (xz)", "γ₁₂ (xy)"]
    labels_row = ["σ₁₁ (xx)", "σ₂₂ (yy)", "σ₃₃ (zz)", "τ₂₃ (yz)", "τ₁₃ (xz)", "τ₁₂ (xy)"]

    # Spójna skala dla macierzy referencyjnej i przewidywanej
    z_min_val = min(float(C_true.min()), float(C_pred.min()))
    z_max_val = max(float(C_true.max()), float(C_pred.max()))

    fig_matrices = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "<b>1. Macierz Wzorcowa (Ground Truth / FEM)</b>",
            "<b>2. Predykcja Sieci Neuronowej (NN)</b>",
            "<b>3. Błąd Bezwzględny |C_true - C_pred|</b>",
        ),
        horizontal_spacing=0.08,
    )

    # 1. Ground Truth
    fig_matrices.add_trace(
        go.Heatmap(
            z=C_true,
            x=labels,
            y=labels_row,
            colorscale="Viridis",
            zmin=z_min_val,
            zmax=z_max_val,
            text=np.round(C_true, 2),
            texttemplate="%{text}",
            textfont=dict(size=10),
            coloraxis="coloraxis1",
        ),
        row=1,
        col=1,
    )

    # 2. NN Prediction
    fig_matrices.add_trace(
        go.Heatmap(
            z=C_pred,
            x=labels,
            y=labels_row,
            colorscale="Viridis",
            zmin=z_min_val,
            zmax=z_max_val,
            text=np.round(C_pred, 2),
            texttemplate="%{text}",
            textfont=dict(size=10),
            coloraxis="coloraxis1",
        ),
        row=1,
        col=2,
    )

    # 3. Absolute Error
    fig_matrices.add_trace(
        go.Heatmap(
            z=C_diff,
            x=labels,
            y=labels_row,
            colorscale="Reds",
            zmin=0.0,
            zmax=max(float(C_diff.max()), 1.0),
            text=np.round(C_diff, 2),
            texttemplate="%{text}",
            textfont=dict(size=10),
            coloraxis="coloraxis2",
        ),
        row=1,
        col=3,
    )

    fig_matrices.update_layout(
        height=520,
        margin=dict(l=40, r=40, b=40, t=60),
        coloraxis1=dict(
            colorscale="Viridis",
            colorbar=dict(title="C [GPa]", x=0.64, len=0.85, thickness=15),
        ),
        coloraxis2=dict(
            colorscale="Reds",
            colorbar=dict(title="|ΔC| [GPa]", x=1.02, len=0.85, thickness=15),
        ),
        yaxis=dict(autorange="reversed"),
        yaxis2=dict(autorange="reversed"),
        yaxis3=dict(autorange="reversed"),
    )

    st.plotly_chart(fig_matrices, width="stretch")

    # Tabela analityczna wartości głównych
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### 🔍 Porównanie Składowych Osiowych (Ściskanie / Rozciąganie)")
        diag_df = pd.DataFrame(
            {
                "Komponent": ["C11 (Oś X)", "C22 (Oś Y)", "C33 (Oś Z)"],
                "Ground Truth [GPa]": [C_true[0, 0], C_true[1, 1], C_true[2, 2]],
                "Predykcja NN [GPa]": [C_pred[0, 0], C_pred[1, 1], C_pred[2, 2]],
                "Błąd Bezwzględny [GPa]": [C_diff[0, 0], C_diff[1, 1], C_diff[2, 2]],
                "Błąd Względny [%]": [
                    (C_diff[0, 0] / C_true[0, 0]) * 100,
                    (C_diff[1, 1] / C_true[1, 1]) * 100,
                    (C_diff[2, 2] / C_true[2, 2]) * 100,
                ],
            }
        )
        st.dataframe(diag_df.style.format({"Ground Truth [GPa]": "{:.2f}", "Predykcja NN [GPa]": "{:.2f}", "Błąd Bezwzględny [GPa]": "{:.2f}", "Błąd Względny [%]": "{:.2f}%"}), width="stretch")

    with col_t2:
        st.markdown("#### ✂️ Porównanie Składowych Ścinania")
        shear_df = pd.DataFrame(
            {
                "Komponent": ["C44 (YZ)", "C55 (XZ)", "C66 (XY)"],
                "Ground Truth [GPa]": [C_true[3, 3], C_true[4, 4], C_true[5, 5]],
                "Predykcja NN [GPa]": [C_pred[3, 3], C_pred[4, 4], C_pred[5, 5]],
                "Błąd Bezwzględny [GPa]": [C_diff[3, 3], C_diff[4, 4], C_diff[5, 5]],
                "Błąd Względny [%]": [
                    (C_diff[3, 3] / C_true[3, 3]) * 100,
                    (C_diff[4, 4] / C_true[4, 4]) * 100,
                    (C_diff[5, 5] / C_true[5, 5]) * 100,
                ],
            }
        )
        st.dataframe(shear_df.style.format({"Ground Truth [GPa]": "{:.2f}", "Predykcja NN [GPa]": "{:.2f}", "Błąd Bezwzględny [GPa]": "{:.2f}", "Błąd Względny [%]": "{:.2f}%"}), width="stretch")

# ----------------------------------------------------
# TAB 3: Kierunkowy Moduł Younga 3D (Anizotropia)
# ----------------------------------------------------
with tab_young:
    st.subheader(f"Trójwymiarowy Wykres Polarny Modułu Younga E(u) ({active_label_map[current_tpms]}, {current_mode.upper()})")
    st.markdown(
        """
        Dla dowolnego wektora kierunkowego $\\mathbf{u} = [l, m, n]^T$, kierunkowy moduł Younga wynika z tensora podatności czwartego rzędu:
        $$\\frac{1}{E(\\mathbf{u})} = S_{ijkl} u_i u_j u_k u_l, \\quad \\text{gdzie } \\mathbf{S} = \\mathbf{C}^{-1}$$
        Kształt trójwymiarowej powierzchni ilustruje anizotropię: dla idealnego materiału izotropowego otrzymujemy sferę, 
        podczas gdy wydłużenie komórek powoduje powstawanie charakterystycznych płatów sztywności.
        """
    )

    col_y_ctrl, col_y_plot = st.columns([1, 3])

    with col_y_ctrl:
        which_mat = st.radio(
            "Wizualizuj na podstawie:",
            ["Predykcja Sieci Neuronowej (NN)", "Wzorzec (Ground Truth)"],
            index=0,
        )
        young_cmap = st.selectbox("Kolorystyka wykresu Younga:", ["Plasma", "Thermal", "Turbo", "Spectral"])
        
        # Wybór macierzy
        active_C = C_pred if which_mat == "Predykcja Sieci Neuronowej (NN)" else C_true

        # Obliczenie powierzchni
        X_y, Y_y, Z_y, E_vals = get_cached_young_surface(active_C, num_theta=45, num_phi=90)

        E_min_val = float(np.min(E_vals))
        E_max_val = float(np.max(E_vals))
        anisotropy_index = E_max_val / max(E_min_val, 1e-6)

        st.markdown(
            f"""
            <div class="formula-box">
                <b>Wskaźniki Sprężystości:</b><br>
                • E_min = <b>{E_min_val:.2f} GPa</b><br>
                • E_max = <b>{E_max_val:.2f} GPa</b><br>
                • Średni E = <b>{float(np.mean(E_vals)):.2f} GPa</b><br>
                • Wskaźnik Anizotropii (E_max / E_min): <b>{anisotropy_index:.2f}</b><br>
                <i>(Wartość 1.0 oznacza idealną izotropię)</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_y_plot:
        fig_young = go.Figure(
            data=go.Surface(
                x=X_y,
                y=Y_y,
                z=Z_y,
                surfacecolor=E_vals,
                colorscale=young_cmap,
                colorbar=dict(title="E(u) [GPa]", len=0.8, thickness=16),
            )
        )

        fig_young.update_layout(
            title=f"Kierunkowa Powierzchnia Sztywności 3D ({which_mat})",
            scene=dict(
                xaxis=dict(title="Ex [GPa]"),
                yaxis=dict(title="Ey [GPa]"),
                zaxis=dict(title="Ez [GPa]"),
                aspectmode="data",
                camera=dict(eye=dict(x=1.7, y=1.7, z=1.3)),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=600,
        )
        st.plotly_chart(fig_young, width="stretch")

# ----------------------------------------------------
# TAB 4: Diagnostyka Uczenia Sieci Neuronowej
# ----------------------------------------------------
with tab_training:
    st.subheader("Historia Uczenia i Ewaluacja Modelu MLP (PyTorch)")

    history = st.session_state.model.history
    if len(history.get("train_loss", [])) > 0:
        epochs_arr = list(range(1, len(history["train_loss"]) + 1))

        col_t_plot1, col_t_plot2 = st.columns(2)

        with col_t_plot1:
            fig_loss = go.Figure()
            fig_loss.add_trace(
                go.Scatter(
                    x=epochs_arr,
                    y=history["train_loss"],
                    mode="lines",
                    name="Train Loss (MSE)",
                    line=dict(color="#38bdf8", width=2.5),
                )
            )
            fig_loss.add_trace(
                go.Scatter(
                    x=epochs_arr,
                    y=history["val_loss"],
                    mode="lines",
                    name="Validation Loss (MSE)",
                    line=dict(color="#f43f5e", width=2.5, dash="dash"),
                )
            )
            fig_loss.update_layout(
                title="Krzywa Funkcji Straty (Loss Curve)",
                xaxis_title="Epoka",
                yaxis_title="MSE Loss",
                yaxis_type="log",
                hovermode="x unified",
                legend=dict(x=0.65, y=0.95),
                height=420,
            )
            st.plotly_chart(fig_loss, width="stretch")

        with col_t_plot2:
            fig_r2 = go.Figure()
            fig_r2.add_trace(
                go.Scatter(
                    x=epochs_arr,
                    y=history["val_r2"],
                    mode="lines+markers",
                    name="Walidacyjne R²",
                    line=dict(color="#10b981", width=2.5),
                    marker=dict(size=4),
                )
            )
            fig_r2.update_layout(
                title="Współczynnik Determinacji R² na Zbiorze Walidacyjnym",
                xaxis_title="Epoka",
                yaxis_title="R² Score",
                yaxis_range=[max(0.0, min(history["val_r2"]) - 0.05), 1.02],
                hovermode="x unified",
                legend=dict(x=0.65, y=0.15),
                height=420,
            )
            st.plotly_chart(fig_r2, width="stretch")

        # Architektura modelu
        st.markdown("#### 📐 Szczegóły Architektury Sieci MLP")
        arch_col1, arch_col2, arch_col3 = st.columns(3)
        with arch_col1:
            st.markdown(
                """
                - **Warstwa Wejściowa:** 4 cechy `[nx, ny, nz, t]`
                - **Normalizacja:** `StandardScaler` (Mean=0, Std=1)
                - **Optymalizator:** AdamW (Weight Decay=1e-4)
                """
            )
        with arch_col2:
            st.markdown(
                f"""
                - **Warstwy Ukryte:** `{st.session_state.model.hidden_dims}`
                - **Aktywacje:** `{st.session_state.model.activation.upper()}` + BatchNorm1d
                - **Scheduler:** `ReduceLROnPlateau` (patience=15)
                """
            )
        with arch_col3:
            st.markdown(
                f"""
                - **Warstwa Wyjściowa:** 21 cech (górny trójkąt Voigta)
                - **Rekonstrukcja:** $C = C^T$ (symetryczna $6 \\times 6$)
                - **Końcowe R²:** `{history['val_r2'][-1]:.4f}`
                """
            )
    else:
        st.info("Brak historii treningu. Kliknij przycisk 'Trenuj Sieć Neuronową' w panelu bocznym.")

# ----------------------------------------------------
# TAB 5: Zbiór Danych & Import/Eksport
# ----------------------------------------------------
with tab_dataset:
    st.subheader("Zarządzanie Zbiorem Danych Treningowych")
    st.markdown(
        "Możesz wygenerować nowy syntetyczny zbiór danych, pobrać go jako plik `.csv` lub `.npz`, albo załadować własny zbiór pomiarowy/symulacyjny MES."
    )

    X_data = st.session_state.dataset_X
    y_data = st.session_state.dataset_y

    col_ds1, col_ds2 = st.columns([1, 1])

    with col_ds1:
        st.markdown(f"#### 📊 Aktualny Zbiór Danych ({X_data.shape[0]} próbek)")
        
        # Tworzenie czytelnej ramki DataFrame
        cols_names = ["nx", "ny", "nz", "t"] + [f"C_tri_{i+1}" for i in range(21)]
        df_preview = pd.DataFrame(np.hstack([X_data, y_data]), columns=cols_names)
        st.dataframe(df_preview.head(10), width="stretch")

        # Eksport do CSV
        csv_buffer = io.StringIO()
        df_preview.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Pobierz zbiór danych jako CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{current_tpms}_stiffness_dataset.csv",
            mime="text/csv",
        )

        # Eksport do NPZ
        npz_buffer = io.BytesIO()
        np.savez_compressed(npz_buffer, X=X_data, y=y_data)
        st.download_button(
            label="📥 Pobierz zbiór danych jako NPZ",
            data=npz_buffer.getvalue(),
            file_name=f"{current_tpms}_stiffness_dataset.npz",
            mime="application/octet-stream",
        )

    with col_ds2:
        st.markdown("#### 📤 Załaduj Własny Zbiór Danych")
        uploaded_file = st.file_uploader(
            "Wybierz plik z danymi (.csv lub .npz):",
            type=["csv", "npz"],
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_uploaded = pd.read_csv(uploaded_file)
                    if df_uploaded.shape[1] >= 25:
                        X_up = df_uploaded.iloc[:, :4].to_numpy(dtype=np.float32)
                        y_up = df_uploaded.iloc[:, 4:25].to_numpy(dtype=np.float32)
                        st.session_state.dataset_X = X_up
                        st.session_state.dataset_y = y_up
                        st.success(f"Pomyślnie załadowano plik CSV z {X_up.shape[0]} próbkami!")
                    else:
                        st.error("Plik CSV musi zawierać co najmniej 25 kolumn ([nx, ny, nz, t] + 21 składowych sztywności).")
                elif uploaded_file.name.endswith(".npz"):
                    loaded_npz = np.load(uploaded_file)
                    if "X" in loaded_npz and "y" in loaded_npz:
                        st.session_state.dataset_X = loaded_npz["X"].astype(np.float32)
                        st.session_state.dataset_y = loaded_npz["y"].astype(np.float32)
                        st.success(f"Pomyślnie załadowano plik NPZ z {loaded_npz['X'].shape[0]} próbkami!")
                    else:
                        st.error("Plik NPZ musi zawierać tablice o kluczach 'X' oraz 'y'.")
            except Exception as e:
                st.error(f"Błąd podczas wczytywania pliku: {e}")

        st.markdown("---")
        st.markdown("#### 💾 Zapis i Wczytanie Wag Modelu")
        model_save_col1, model_save_col2 = st.columns(2)
        with model_save_col1:
            if st.button("Zapisz wagi do pliku .pth"):
                st.session_state.model.save_checkpoint("gyroid_nn_checkpoint.pth")
                st.success("Zapisano model do `gyroid_nn_checkpoint.pth`!")
        with model_save_col2:
            if os.path.exists("gyroid_nn_checkpoint.pth"):
                if st.button("Wczytaj wagi z pliku .pth"):
                    st.session_state.model.load_checkpoint("gyroid_nn_checkpoint.pth")
                    st.success("Wczytano model z pliku!")

    # ========================================================
    # Sekcja Eksportu Modeli 3D (CAD: STEP, IGES, STL, OBJ, PLY)
    # ========================================================
    st.markdown("---")
    st.subheader("📐 Generator i Eksport Modeli 3D (CAD & Additive Manufacturing)")
    st.markdown(
        """
        Generuj i pobieraj trójwymiarowe modele struktur TPMS i metamateriałów auksetycznych w formatach inżynierskich:
        - **STEP (`.step` / `.stp`)**: Standard ISO 10303 dla systemów CAD (SolidWorks, Autodesk Inventor, Fusion 360, CATIA, Siemens NX, FreeCAD).
        - **IGES (`.iges` / `.igs`)**: Klasyczny standard ANSI Y14.26M geometrii powierzchniowej.
        - **STL (`.stl`)**: Standard triangulacji dla druku 3D (Slicery: PrusaSlicer, Bambu Studio, Cura, OrcaSlicer).
        - **OBJ (`.obj`) / PLY (`.ply`)**: Otwarte formaty siatek poligonowych dla grafiki i inżynierii odwrotnej.
        """
    )

    cad_col1, cad_col2 = st.columns([1, 1])

    with cad_col1:
        st.markdown("##### ⚙️ Parametry Geometryczne Modelu 3D")
        export_tpms_type = st.radio(
            "Struktura metamateriału do wyeksportowania:",
            [
                "🧊 Schoen Gyroid (G)",
                "💎 Schwarz Diamond (D)",
                "📦 Schwarz Primitive (P)",
                "🎀 Re-entrant Auxetic (ν < 0)",
                "🕸️ Neovius (Schoen C(P))",
                "🔀 Schoen I-WP (Asymetryczny)",
                "🏗️ Octet-Truss (Schoen F-RD)",
            ],
            index=default_idx,
            horizontal=True,
            key="cad_export_tpms_type_sel",
        )
        if "Diamond" in export_tpms_type:
            selected_export_type = "diamond"
        elif "Primitive" in export_tpms_type:
            selected_export_type = "primitive"
        elif "Auxetic" in export_tpms_type:
            selected_export_type = "auxetic"
        elif "Neovius" in export_tpms_type:
            selected_export_type = "neovius"
        elif "I-WP" in export_tpms_type:
            selected_export_type = "iwp"
        elif "Octet" in export_tpms_type:
            selected_export_type = "octet"
        else:
            selected_export_type = "gyroid"

        cad_mode_choice = st.radio(
            "Morfologia eksportu:",
            ["🧱 Network (Lity szkielet)", "📜 Sheet (Ścianka powłokowa)"],
            index=0 if current_mode == "network" else 1,
            horizontal=True,
            key="cad_export_mode_sel",
        )
        selected_export_mode = "sheet" if "Sheet" in cad_mode_choice else "network"

        if selected_export_mode == "sheet":
            cad_tw = st.slider("Grubość ścianki tw [0.05 - 0.45]:", 0.05, 0.45, float(wall_thickness_val), 0.01, key="cad_tw_slider")
        else:
            cad_tw = 0.20

        export_format = st.selectbox(
            "Format pliku docelowego:",
            [
                "STEP (.step) — Model CAD (ISO 10303 Faceted B-Rep)",
                "IGES (.iges) — Powierzchnia CAD (ANSI Y14.26M)",
                "STL (.stl) — Siatka do Druku 3D (Binarny)",
                "OBJ (.obj) — Siatka Wavefront",
                "PLY (.ply) — Format Polygon Stanford",
            ],
            index=0,
            key="cad_export_fmt_select",
        )

        mesh_density_choice = st.select_slider(
            "Dokładność / Rozdzielczość próbkowania siatki:",
            options=["Szybka / CAD (18³)", "Standardowa (22³)", "Wysoka / Druk 3D (26³)", "Bardzo wysoka (30³)"],
            value="Standardowa (22³)",
            key="cad_export_mesh_res",
        )
        res_map = {"Szybka / CAD (18³)": 18, "Standardowa (22³)": 22, "Wysoka / Druk 3D (26³)": 26, "Bardzo wysoka (30³)": 30}
        chosen_res = res_map[mesh_density_choice]

        dim_col1, dim_col2, dim_col3 = st.columns(3)
        with dim_col1:
            size_x = st.number_input("Wymiar X [mm]:", min_value=5.0, max_value=200.0, value=25.0, step=5.0, key="cad_size_x")
        with dim_col2:
            size_y = st.number_input("Wymiar Y [mm]:", min_value=5.0, max_value=200.0, value=25.0, step=5.0, key="cad_size_y")
        with dim_col3:
            size_z = st.number_input("Wymiar Z [mm]:", min_value=5.0, max_value=200.0, value=25.0, step=5.0, key="cad_size_z")

    with cad_col2:
        st.markdown("##### 🛠️ Generowanie i Pobieranie Pliku")
        st.markdown(
            f"""
            <div class="formula-box">
                <b>Parametry eksportowanego bloku:</b><br>
                • Struktura: <b>{active_label_map[selected_export_type]}</b><br>
                • Morfologia: <b>{'📜 Sheet (Ścianka)' if selected_export_mode == 'sheet' else '🧱 Network (Bryła)'}</b><br>
                • Liczba komórek: <b>nx={nx:.2f}, ny={ny:.2f}, nz={nz:.2f}</b><br>
                • Wartość progu t: <b>{t_val:.2f}</b>{' (tw=' + str(round(cad_tw, 2)) + ')' if selected_export_mode == 'sheet' else ''}<br>
                • Wymiary gabarytowe: <b>{size_x:.1f} × {size_y:.1f} × {size_z:.1f} mm</b><br>
                • Rozdzielczość siatki: <b>{chosen_res}³ = {chosen_res**3:,}</b> wokseli
            </div>
            """,
            unsafe_allow_html=True,
        )

        gen_cad_btn = st.button("🚀 Wygeneruj i Przygotuj Model 3D", type="primary", width="stretch", key="cad_gen_button")

        if gen_cad_btn:
            with st.spinner("Przetwarzanie geometrii metamateriału algorytmem Marching Cubes oraz konwersja do formatu CAD..."):
                try:
                    mesh_3d = exp3d.generate_tpms_mesh(
                        tpms_type=selected_export_type,
                        nx=nx,
                        ny=ny,
                        nz=nz,
                        t=t_val,
                        resolution=chosen_res,
                        size_x_mm=float(size_x),
                        size_y_mm=float(size_y),
                        size_z_mm=float(size_z),
                        mode=selected_export_mode,
                        wall_thickness=cad_tw,
                    )

                    num_v = len(mesh_3d.vertices)
                    num_f = len(mesh_3d.faces)

                    if "STEP" in export_format:
                        file_bytes = exp3d.export_to_step(mesh_3d)
                        ext = "step"
                        mime_type = "application/step"
                    elif "IGES" in export_format:
                        file_bytes = exp3d.export_to_iges(mesh_3d)
                        ext = "iges"
                        mime_type = "application/iges"
                    elif "STL" in export_format:
                        file_bytes = exp3d.export_to_stl(mesh_3d, binary=True)
                        ext = "stl"
                        mime_type = "application/sla"
                    elif "OBJ" in export_format:
                        file_bytes = exp3d.export_to_obj(mesh_3d)
                        ext = "obj"
                        mime_type = "text/plain"
                    else:
                        file_bytes = exp3d.export_to_ply(mesh_3d)
                        ext = "ply"
                        mime_type = "application/octet-stream"

                    file_size_kb = len(file_bytes) / 1024.0

                    st.session_state["last_cad_export"] = {
                        "bytes": file_bytes,
                        "filename": f"{selected_export_type}_{selected_export_mode}_{int(size_x)}x{int(size_y)}x{int(size_z)}mm.{ext}",
                        "mime": mime_type,
                        "ext": ext,
                        "num_v": num_v,
                        "num_f": num_f,
                        "size_kb": file_size_kb,
                        "label": f"{selected_export_type.upper()}_{selected_export_mode.upper()}_{int(size_x)}mm.{ext}",
                    }
                except Exception as e:
                    st.error(f"⚠️ Nie udało się wygenerować modelu 3D: {e}")

        if "last_cad_export" in st.session_state:
            cad_exp = st.session_state["last_cad_export"]
            st.success(
                f"Model **.{cad_exp['ext'].upper()}** gotowy do pobrania!\n\n"
                f"• Wierzchołki: **{cad_exp['num_v']:,}** | Trójkąty: **{cad_exp['num_f']:,}** | Rozmiar pliku: **{cad_exp['size_kb']:.1f} KB**"
            )
            st.download_button(
                label=f"💾 Pobierz plik {cad_exp['label']}",
                data=cad_exp["bytes"],
                file_name=cad_exp["filename"],
                mime=cad_exp["mime"],
                width="stretch",
                key="cad_download_final_btn",
            )

# Stopka aplikacji
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
        Gyroid Metamaterial Stiffness NN Studio • Mechanika Obliczeniowa & TPMS • PyTorch + Streamlit + Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
