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

import tpms_generator as tg
import nn_model as nm

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
@st.cache_data(show_spinner=False)
def get_cached_gyroid_grid(nx: float, ny: float, nz: float, t: float, resolution: int):
    """Generowanie siatki z pamięcią podręczną."""
    return tg.generate_gyroid_grid(nx, ny, nz, t, resolution=resolution)


@st.cache_data(show_spinner=False)
def get_cached_synthetic_data(num_samples: int, noise_level: float, seed: int):
    """Generowanie danych syntetycznych z pamięcią podręczną."""
    return tg.generate_synthetic_dataset(num_samples=num_samples, noise_level=noise_level, random_seed=seed)


@st.cache_data(show_spinner=False)
def get_cached_young_surface(C: np.ndarray, num_theta: int = 40, num_phi: int = 80):
    """Obliczanie kierunkowego modułu Younga z pamięcią podręczną."""
    return tg.calculate_directional_youngs_modulus(C, num_theta=num_theta, num_phi=num_phi)


def init_session_state():
    """Inicjalizacja modelu sieci i domyślnych danych w st.session_state."""
    if "model" not in st.session_state:
        st.session_state.model = nm.StiffnessSurrogateModel(
            hidden_dims=[64, 128, 128, 64],
            lr=0.003,
            activation="gelu",
        )

    if "dataset_X" not in st.session_state or "dataset_y" not in st.session_state:
        X_init, y_init = tg.generate_synthetic_dataset(num_samples=500, random_seed=42)
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
        <h1>🧊 Gyroid Metamaterial Stiffness NN Studio</h1>
        <p>Zaawansowane modelowanie i wizualizacja relacji mikromechanicznych TPMS (Gyroid) 
        oraz aproksymacja macierzy sztywności 6×6 (Voigt) z użyciem głębokich sieci neuronowych PyTorch.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# Pasek boczny (Sidebar) - Sterowanie
# ==========================================
with st.sidebar:
    st.header("⚙️ Parametry Geometryczne Gyroidu")

    # Presety parametrów
    preset_choice = st.selectbox(
        "Wybierz predefiniowaną konfigurację (Preset):",
        [
            "Własna konfiguracja (Custom)",
            "Izotropowy standardowy (nx=2, ny=2, nz=2, t=0)",
            "Anizotropowy belkowy (nx=4.5, ny=1.5, nz=1.5, t=0.1)",
            "Płytowy / Warstwowy (nx=1.5, ny=1.5, nz=4.5, t=-0.2)",
            "Wysoka porowatość (t=-0.7)",
            "Gęsta struktura ściankowa (t=0.6)",
        ],
    )

    # Wartości domyślne na podstawie presetu
    d_nx, d_ny, d_nz, d_t = 2.0, 2.0, 2.0, 0.0
    if preset_choice == "Izotropowy standardowy (nx=2, ny=2, nz=2, t=0)":
        d_nx, d_ny, d_nz, d_t = 2.0, 2.0, 2.0, 0.0
    elif preset_choice == "Anizotropowy belkowy (nx=4.5, ny=1.5, nz=1.5, t=0.1)":
        d_nx, d_ny, d_nz, d_t = 4.5, 1.5, 1.5, 0.1
    elif preset_choice == "Płytowy / Warstwowy (nx=1.5, ny=1.5, nz=4.5, t=-0.2)":
        d_nx, d_ny, d_nz, d_t = 1.5, 1.5, 4.5, -0.2
    elif preset_choice == "Wysoka porowatość (t=-0.7)":
        d_nx, d_ny, d_nz, d_t = 2.5, 2.5, 2.5, -0.7
    elif preset_choice == "Gęsta struktura ściankowa (t=0.6)":
        d_nx, d_ny, d_nz, d_t = 2.5, 2.5, 2.5, 0.6

    nx = st.slider("Liczba komórek nx (Oś X):", min_value=1.0, max_value=5.0, value=float(d_nx), step=0.1)
    ny = st.slider("Liczba komórek ny (Oś Y):", min_value=1.0, max_value=5.0, value=float(d_ny), step=0.1)
    nz = st.slider("Liczba komórek nz (Oś Z):", min_value=1.0, max_value=5.0, value=float(d_nz), step=0.1)
    t_val = st.slider("Parametr izopowierzchni t (Gęstość):", min_value=-1.0, max_value=1.0, value=float(d_t), step=0.05)

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

    train_btn = st.button("🚀 Trenuj Sieć Neuronową", use_container_width=True, type="primary")

    st.markdown("---")
    mesh_res = st.select_slider("Rozdzielczość siatki 3D Gyroidu:", options=[24, 30, 36, 42], value=30)
    
    st.info(
        f"**Status modelu:** {'✅ Wytrenowany' if st.session_state.model.is_trained else '⏳ Oczekuje na trening'}\n\n"
        f"**Urządzenie PyTorch:** `{st.session_state.model.device}`"
    )

# Obsługa przycisku treningu
if train_btn:
    with st.spinner("Generowanie nowych danych syntetycznych i trening sieci MLP..."):
        X_new, y_new = tg.generate_synthetic_dataset(
            num_samples=num_samples,
            random_seed=int(time.time()),
            noise_level=noise_param,
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
C_true = tg.synthetic_stiffness_matrix(nx, ny, nz, t_val, noise_level=0.0)

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
rho_calc = tg.compute_relative_density(nx, ny, nz, t_val, resolution=25)

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
        "🧊 1. Wizualizacja 3D Gyroidu (TPMS)",
        "📊 2. Macierze Sztywności 6×6 (Porównanie)",
        "🌐 3. Kierunkowy Moduł Younga 3D (Anizotropia)",
        "📈 4. Diagnostyka Uczenia Sieci",
        "💾 5. Zbiór Danych & Eksport/Import",
    ]
)

# ----------------------------------------------------
# TAB 1: Wizualizacja 3D Struktury Gyroidu
# ----------------------------------------------------
with tab_3d_geom:
    st.subheader("Trójwymiarowy Model Powierzchni TPMS Gyroidu")
    st.markdown(
        """
        Równanie powierzchni:
        $$f(x, y, z) = \\sin(k_x x)\\cos(k_y y) + \\sin(k_y y)\\cos(k_z z) + \\sin(k_z z)\\cos(k_x x) - t = 0$$
        gdzie $k_i = 2\\pi n_i / L_i$. Powierzchnia dzieli przestrzeń na labiryntowe domeny komórkowe o zerowej średniej krzywiźnie.
        """
    )

    col_iso_opt1, col_iso_opt2 = st.columns([1, 3])
    with col_iso_opt1:
        color_scheme = st.selectbox(
            "Paleta barw:",
            ["Viridis", "Plasma", "Cividis", "Turbo", "Blues", "Magma"],
            index=0,
        )
        opacity_val = st.slider("Przezroczystość powierzchni:", 0.3, 1.0, 0.95, 0.05)
        show_caps = st.checkbox("Zamknij ściany komórki (Caps)", value=False)

        st.markdown(
            f"""
            <div class="formula-box">
                <b>Aktualne parametry TPMS:</b><br>
                • nx = <b>{nx:.2f}</b>, ny = <b>{ny:.2f}</b>, nz = <b>{nz:.2f}</b><br>
                • Wartość progu t = <b>{t_val:.2f}</b><br>
                • Gęstość względna ρ ≈ <b>{rho_calc:.3f}</b><br>
                • Siatka 3D: <b>{mesh_res}³ = {mesh_res**3:,}</b> punktów
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_iso_opt2:
        # Generowanie siatki gyroidu
        X_grid, Y_grid, Z_grid, F_grid = get_cached_gyroid_grid(nx, ny, nz, t_val, resolution=mesh_res)

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
                colorbar=dict(title="f(x,y,z)", thickness=15, len=0.7),
            )
        )

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
        st.plotly_chart(fig_iso, use_container_width=True)

# ----------------------------------------------------
# TAB 2: Macierze Sztywności 6x6 (Heatmaps)
# ----------------------------------------------------
with tab_matrix:
    st.subheader("Zestawienie Macierzy Sztywności 6×6 w Notacji Voigta (w GPa)")
    st.markdown(
        "Porównanie rzeczywistej ortotropowej macierzy sztywności ($C_{true}$), predykcji sieci neuronowej ($C_{pred}$) oraz bezwzględnej macierzy różnic $|C_{true} - C_{pred}|$."
    )

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

    st.plotly_chart(fig_matrices, use_container_width=True)

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
        st.dataframe(diag_df.style.format({"Ground Truth [GPa]": "{:.2f}", "Predykcja NN [GPa]": "{:.2f}", "Błąd Bezwzględny [GPa]": "{:.2f}", "Błąd Względny [%]": "{:.2f}%"}), use_container_width=True)

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
        st.dataframe(shear_df.style.format({"Ground Truth [GPa]": "{:.2f}", "Predykcja NN [GPa]": "{:.2f}", "Błąd Bezwzględny [GPa]": "{:.2f}", "Błąd Względny [%]": "{:.2f}%"}), use_container_width=True)

# ----------------------------------------------------
# TAB 3: Kierunkowy Moduł Younga 3D (Anizotropia)
# ----------------------------------------------------
with tab_young:
    st.subheader("Trójwymiarowy Wykres Polarny Modułu Younga E(u) (Anizotropia Sprężysta)")
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
        st.plotly_chart(fig_young, use_container_width=True)

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
            st.plotly_chart(fig_loss, use_container_width=True)

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
            st.plotly_chart(fig_r2, use_container_width=True)

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
        st.dataframe(df_preview.head(10), use_container_width=True)

        # Eksport do CSV
        csv_buffer = io.StringIO()
        df_preview.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Pobierz zbiór danych jako CSV",
            data=csv_buffer.getvalue(),
            file_name="gyroid_stiffness_dataset.csv",
            mime="text/csv",
        )

        # Eksport do NPZ
        npz_buffer = io.BytesIO()
        np.savez_compressed(npz_buffer, X=X_data, y=y_data)
        st.download_button(
            label="📥 Pobierz zbiór danych jako NPZ",
            data=npz_buffer.getvalue(),
            file_name="gyroid_stiffness_dataset.npz",
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
