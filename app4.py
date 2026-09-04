import streamlit as st
import numpy as np
import tensorflow as tf
tf.compat.v1.reset_default_graph()
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import keras
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import sys
# ============================================================
# PyTorch availability
# ============================================================

TORCH_AVAILABLE = False
torch = None

try:
    import torch
    TORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__} loaded successfully.")
except Exception as e:
    print(f"WARNING: PyTorch unavailable: {e}")

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
T_MIN, T_MAX  = 376.0, 2700.1
T_SIM_MAX     = 7200.0
X_MAX, Z_MAX  = 4.4, 5.0
NZ, NX        = 101, 89
PIXEL_SIZE    = 0.05
N_FRAMES      = 60      # number of time frames to generate
MFUEL         = 136000.0

# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════
def compute_geometry(Mfuel, Ffuel, Fstruct, Porosity, Rflat, Alpha):
    Pi = 4.0 * np.arctan(1.0)
    Mcorium = Mfuel * Ffuel * (1 + Fstruct)
    Vdebris = Mcorium / 7300.0 / (1.0 - Porosity)
    Rcav, Hcav, Hwater = 4.4, 5.0, 5.0
    Rlow, Hlow = 3.35, 1.0
    alpha = Alpha * Pi / 180.0
    tana  = np.tan(alpha)

    V1 = Pi / 3.0 * tana * (Rlow**3 - Rflat**3)
    V2 = V1 + Pi * Hlow * Rlow**2
    V3 = V2 - V1 + Pi / 3.0 * tana * (Rcav**3 - Rflat**3)

    if Vdebris < V1:
        R = (Rflat**3 + 3.0 * Vdebris / (Pi * tana))**(1.0/3.0)
        H = tana * (R - Rflat)
        rDebris = [0.0, R, Rflat, 0.0]
        zDebris = [0.0, 0.0, H, H]
    elif Vdebris < V2:
        Vcyl = Vdebris - V1
        Hcyl = Vcyl / (Pi * Rlow**2)
        Hcon = tana * (Rlow - Rflat)
        H    = Hcyl + Hcon
        rDebris = [0.0, Rlow, Rlow, Rflat, 0.0]
        zDebris = [0.0, 0.0, Hcyl, H, H]
    elif Vdebris < V3:
        Vcon = Vdebris - Pi * Hlow * Rlow**2
        R    = (Rflat**3 + 3.0 * Vcon / (Pi * tana))**(1.0/3.0)
        Hcon = tana * (R - Rflat)
        H    = Hlow + Hcon
        rDebris = [0.0, Rlow, Rlow, R, Rflat, 0.0]
        zDebris = [0.0, 0.0, Hlow, Hlow, H, H]
    else:
        Vcyl = Vdebris - V3
        Hcyl = Vcyl / (Pi * Rcav**2) + Hlow
        Hcon = tana * (Rcav - Rflat)
        H    = Hcyl + Hcon
        rDebris = [0.0, Rlow, Rlow, Rcav, Rcav, Rflat, 0.0]
        zDebris = [0.0, 0.0, Hlow, Hlow, Hcyl, H, H]

    return rDebris, zDebris, H


def build_por_map(Mfuel, Ffuel, Fstruct, Porosity, Rflat, Alpha):
    """
    Rasterise the debris bed geometry onto the 101×89 common grid.
    Returns (1, 101, 89) float32 POR map.
    """
    rDebris, zDebris, _ = compute_geometry(Mfuel, Ffuel, Fstruct,
                                            Porosity, Rflat, Alpha)
    x_grid = np.linspace(0, X_MAX, NX)
    z_grid = np.linspace(0, Z_MAX, NZ)
    Xq, Zq = np.meshgrid(x_grid, z_grid)

    # point-in-polygon for radially symmetric bed (r = x here)
    from matplotlib.path import Path
    polygon = list(zip(rDebris, zDebris))
    path    = Path(polygon)
    pts     = np.column_stack([Xq.ravel(), Zq.ravel()])
    inside  = path.contains_points(pts).reshape(NZ, NX)

    por_map = np.where(inside, float(Porosity), 0.0).astype(np.float32)
    return por_map[np.newaxis, :, :]   # (1, 101, 89)


def plot_geometry(rDebris, zDebris):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_facecolor('lightblue')
    ax.fill_betweenx([0, 5], 0, 4.4, color='lightblue', alpha=0.5, label='Water')
    ax.fill(rDebris, zDebris, facecolor='red', alpha=0.5,
            edgecolor='black', label='Debris Bed')
    ax.fill_betweenx([0, 1], 3.35, 4.4, color='grey', alpha=0.9,
                     label='Structural Block')
    ax.set_xlim(0, 4.4); ax.set_ylim(0, 5)
    ax.set_xlabel("R (m)"); ax.set_ylabel("Z (m)")
    ax.set_aspect("equal"); ax.grid(True); ax.legend()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION / REGRESSION MODELS (existing)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    base_dir = os.path.join(os.path.dirname(__file__), "models_saved")
    models = []
    for i in range(1, 6):
        path = os.path.join(base_dir, f"model_new_model_{i}")
        if os.path.exists(path):
            try:
                models.append(tf.keras.models.load_model(path, compile=False))
            except Exception as e:
                st.warning(f"Failed to load model {path}: {e}")
    return models


def load_classifier():
    return joblib.load("OptimizedVotingClassifier.pkl")


@st.cache_resource
def load_scaler_X():
    return joblib.load("MinMax_scaler_X_Classification.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# ENERGY PINN (existing)
# ═══════════════════════════════════════════════════════════════════════════════
E_0 = 2.983915e11
P_0 = 3.781024e7
T_0 = E_0 / P_0
THERMAL_POWER = 3.84e9

ENERGY_PARAM_RANGES = [
    ("Psys",      0.11e6, 0.5e6),
    ("Ffuel",     0.5,    1.0),
    ("Porosity",  0.25,   0.5),
    ("Dparticle", 0.001,  0.005),
    ("Alpha",     15.0,   45.0),
    ("Rflat",     0.0,    2.0),
    ("Tbed",      400.0,  1700.0),
    ("Decay",     0.003,  0.01),
    ("Fstruct",   0.25,   2.0),
]

ENERGY_OUTPUT_LABELS = [
    r"$E_{pd}$  [MW]", r"$Q_{si}$  [MW]",
    r"$Q_{sg}$  [MW]", r"$Q_{sl}$  [MW]",
]


def build_energy_model():
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(11,)),
        tf.keras.layers.Dense(256, activation="tanh"),
        tf.keras.layers.Dense(256, activation="tanh"),
        tf.keras.layers.Dense(256, activation="tanh"),
        tf.keras.layers.Dense(128, activation="tanh"),
        tf.keras.layers.Dense(4),
    ])
    return m


def _load_weights_from_h5(model, path):
    import h5py
    keys = [
        "hidden_layers/dense",   "hidden_layers/dense_1",
        "hidden_layers/dense_2", "hidden_layers/dense_3",
        "layers/dense_4",
    ]
    with h5py.File(path, "r") as f:
        for layer, key in zip(model.layers, keys):
            kernel = f[f"{key}/vars/0"][...]
            bias   = f[f"{key}/vars/1"][...]
            layer.set_weights([kernel, bias])
    return model


@st.cache_resource
def load_energy_ensemble(variant="pinn", n_members=3):
    base = os.path.join(os.path.dirname(__file__), "energy_models")
    members = []
    for i in range(n_members):
        path = os.path.join(base, f"{variant}_member{i}.weights.h5")
        if not os.path.exists(path):
            continue
        try:
            m = build_energy_model()
            _load_weights_from_h5(m, path)
            members.append(m)
        except Exception as e:
            st.warning(f"Failed to load {path}: {e}")
    return members


def build_energy_input(user_inputs_SI, times_s):
    static_star = np.array([
        (val - lo) / (hi - lo)
        for val, (_, lo, hi) in zip(user_inputs_SI, ENERGY_PARAM_RANGES)
    ], dtype=np.float32)
    Ffuel    = user_inputs_SI[1]
    Decay    = user_inputs_SI[7]
    pow_star = (THERMAL_POWER * Decay * Ffuel) / P_0
    n  = len(times_s)
    X  = np.zeros((n, 11), dtype=np.float32)
    X[:, 0]    = np.asarray(times_s) / T_0
    X[:, 1:10] = static_star
    X[:, 10]   = pow_star
    return X


def inverse_scaler_y(value):
    return np.expm1(value)


# ═══════════════════════════════════════════════════════════════════════════════
# UNET V2 SURROGATE
# ═══════════════════════════════════════════════════════════════════════════════
UNET_PARAM_RANGES = {
    'Psys'      : (100000.0, 700000.0),
    'Ffuel'     : (0.3,      1.0),
    'Porosity'  : (0.26,     0.64),
    'Dparticle' : (0.001,    0.006),
    'Alpha'     : (10.0,     45.0),
    'Rflat'     : (0.1,      2.0),
    'Tbed'      : (400.0,    2000.0),
    'Decay'     : (0.002,    0.015),
    'Fstruct'   : (0.1,      1.5),
}
@st.cache_resource
def load_unet_ensemble():
    """Load UNet V2 ensemble."""

    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))

        sys.path.insert(0, app_dir)
        from train_unet_v2 import Config, UNetSurrogate

        cfg = Config()
        device = torch.device("cpu")

        models = []

        for m in range(3):

            ckpt_path = os.path.join(
                app_dir,
                "unet_models",
                f"model_member{m}.pt"
            )

            print("=" * 60)
            print(f"Looking for: {ckpt_path}")
            print(f"Exists: {os.path.isfile(ckpt_path)}")

            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"UNet checkpoint does not exist:\n{ckpt_path}"
                )

            print(f"Loading model {m}...")

            model = UNetSurrogate(cfg).to(device)

            ckpt = torch.load(
                ckpt_path,
                map_location=device,
                weights_only=False
            )

            print(f"Checkpoint type: {type(ckpt)}")
            print(f"Checkpoint keys: {ckpt.keys() if isinstance(ckpt, dict) else 'NOT A DICT'}")

            model.load_state_dict(ckpt["model_state"])

            model.eval()
            models.append(model)

            print(f"Model {m} loaded successfully.")

        print("=" * 60)
        print(f"SUCCESS: loaded {len(models)} UNet models")

        return models, device

    except Exception as e:
        print("=" * 60)
        print("UNET LOADING FAILED")
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)

        st.error(f"UNet loading failed: {type(e).__name__}: {e}")

        return None, None



def normalise_params(user_inputs_SI):
    """Normalise 9 physical params to [0,1] using UNet training ranges."""
    keys = ['Psys','Ffuel','Porosity','Dparticle',
            'Alpha','Rflat','Tbed','Decay','Fstruct']
    vec = []
    for i, k in enumerate(keys):
        lo, hi = UNET_PARAM_RANGES[k]
        vec.append((user_inputs_SI[i] - lo) / (hi - lo))
    return np.array(vec, dtype=np.float32)


@st.cache_data(show_spinner=False)
def predict_temperature_sequence(_models_key, param_tuple, por_map_tuple,
                                  t_end_s, n_frames=N_FRAMES):
    """
    Pre-compute all temperature frames.
    Colour limits computed from all frames so they are consistent.
    """
    unet_models, device = load_unet_ensemble()
    if unet_models is None:
        return None

    # param_tuple must be 9 values already in SI units and correct order:
    # Psys(Pa), Ffuel, Porosity, Dparticle(m), Alpha(deg), Rflat(m),
    # Tbed(K), Decay(fraction), Fstruct
    params_norm = normalise_params(np.array(param_tuple[:9], dtype=np.float32))

    # debug — print normalised values to console
    keys = ['Psys','Ffuel','Porosity','Dparticle','Alpha','Rflat','Tbed','Decay','Fstruct']
    print("=" * 50)
    print("UNET INPUT DEBUG")
    for k, raw, norm in zip(keys, param_tuple[:9], params_norm):
        print(f"  {k:<12}: raw={raw:.4g}  normalised={norm:.4f}")
    print(f"  t_end_s    : {t_end_s:.1f}")
    print("=" * 50)

    por_map    = np.array(por_map_tuple, dtype=np.float32).reshape(1, NZ, NX)
    t_end_norm = t_end_s / T_SIM_MAX
    times_s    = np.linspace(0, t_end_s, n_frames)

    T_mean_all = np.zeros((n_frames, NZ, NX), dtype=np.float32)
    T_std_all  = np.zeros((n_frames, NZ, NX), dtype=np.float32)

    POR_t = torch.from_numpy(por_map[np.newaxis]).to(device)  # (1,1,NZ,NX)

    for i, t_s in enumerate(times_s):
        t_abs_norm = t_s / T_SIM_MAX
        t_rel      = float(np.clip(t_abs_norm / max(t_end_norm, 1e-8), 0, 1))

        # 11-value input: 9 normalised params + t_rel + t_end_norm
        X_t = torch.from_numpy(
            np.concatenate([params_norm,
                            np.array([t_rel, t_end_norm], dtype=np.float32)])
        )[None].to(device)  # (1, 11)

        preds = []
        with torch.no_grad():
            for model in unet_models:
                preds.append(model(X_t, POR_t).cpu().numpy()[0])
        preds = np.stack(preds)
        T_mean_all[i] = preds.mean(0)
        T_std_all[i]  = preds.std(0)

    T_range = T_MAX - T_MIN
    mask    = por_map[0] > 0.01

    # denormalise
    T_K_mean = T_mean_all * T_range + T_MIN
    T_K_std  = T_std_all  * T_range

    # global colour limits from all frames — adapt to this experiment's range
    T_masked = np.where(mask[np.newaxis], T_K_mean, np.nan)
    S_masked = np.where(mask[np.newaxis], T_K_std,  np.nan)
    vmin_global = float(np.nanmin(T_masked))
    vmax_global = float(np.nanmax(T_masked))
    smax_global = float(np.nanmax(S_masked))

    print(f"T range across all frames: [{vmin_global:.1f}, {vmax_global:.1f}] K")
    print(f"σ max across all frames:   {smax_global:.1f} K")

    return {
        'T_mean'  : T_K_mean,
        'T_std'   : T_K_std,
        'times_s' : times_s,
        'por_map' : mask,
        'vmin'    : vmin_global,
        'vmax'    : vmax_global,
        'smax'    : smax_global,
    }


def render_temperature_frame(results, frame_idx):
    """Render prediction + uncertainty for one frame using global colour limits."""

    T_mean = results['T_mean'][frame_idx]
    T_std  = results['T_std'][frame_idx]
    mask   = results['por_map']   # bool (NZ, NX)
    t_s    = results['times_s'][frame_idx]

    # Global colour limits
    vmin = results['vmin']
    vmax = results['vmax']
    smax = results['smax']

    x_grid = np.linspace(0, X_MAX, NX)
    z_grid = np.linspace(0, Z_MAX, NZ)
    Xq, Zq = np.meshgrid(x_grid, z_grid)

    # Larger figure
    fig = plt.figure(figsize=(8, 5))

    # Small gap between subplots
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[20, 1],
        hspace=0.4,
        wspace=0.05
    )
    hours = int(t_s // 3600)
    minutes = int((t_s % 3600) // 60)
    seconds = int(t_s % 60)

    fig.suptitle(
    f't = {hours} h {minutes} min {seconds} s',
    fontsize=25,
    y=1.05
    )
    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    pc1 = ax1.pcolormesh(
        Xq, Zq,
        np.where(mask, T_mean, np.nan),
        cmap='turbo',
        vmin=vmin,
        vmax=vmax,
        shading='auto'
    )

    ax1.set_title(
        'Predicted Temperature (K)',
        fontsize=18
    )
    ax1.set_xlabel('x (m)', fontsize=15)
    ax1.set_ylabel('z (m)', fontsize=15)

    ax1.set_aspect('equal')
    ax1.set_xlim(0, X_MAX)
    ax1.tick_params(labelsize=13)

    # Colour bar
    cax1 = fig.add_subplot(gs[1, 0])
    cb1 = fig.colorbar(
        pc1,
        cax=cax1,
        orientation='horizontal'
    )

    cb1.set_label(
        'Temperature (K)',
        fontsize=15
    )
    cb1.ax.tick_params(labelsize=13)


    # --------------------------------------------------
    # Uncertainty
    # --------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])

    pc2 = ax2.pcolormesh(
        Xq, Zq,
        np.where(mask, T_std, np.nan),
        cmap='viridis',
        vmin=0,
        vmax=smax,
        shading='auto'
    )
    # geometric structural block — fixed geometry, same on every frame
    # block sits from r=3.35 to r=4.4m, z=0 to z=1.0m

    block_r = [3.35, 4.4, 4.4, 3.35]
    block_z = [0.0,  0.0, 1.0, 1.0]

    for ax in [ax1, ax2]:
        block = Polygon(list(zip(block_r, block_z)), closed=True)
        pc_block = PatchCollection([block], facecolor='grey',
                                    edgecolor='darkgrey',
                                    alpha=0.85, zorder=5)
        ax.add_collection(pc_block)

    ax2.set_title(
        'Uncertainty σ (K)',
        fontsize=18
    )
    ax2.set_xlabel('x (m)', fontsize=15)
    ax2.set_ylabel('z (m)', fontsize=15)

    ax2.set_aspect('equal')
    ax2.set_xlim(0, X_MAX)
    ax2.tick_params(labelsize=13)

    # Colour bar
    cax2 = fig.add_subplot(gs[1, 1])
    cb2 = fig.colorbar(
        pc2,
        cax=cax2,
        orientation='horizontal'
    )

    cb2.set_label(
        'Uncertainty σ (K)',
        fontsize=15
    )
    cb2.ax.tick_params(labelsize=13)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLE
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Debris Bed AI", layout="wide")

st.markdown("""
<style>
.section-title {
    border-radius: 10px;
    padding: 0.55rem 0.9rem;
    margin: 0.4rem 0 0.8rem 0;
    font-weight: 700;
    font-size: 1.15rem;
}
.section-input { background: linear-gradient(90deg, #e8f1fb, #f5f9fd); border-left: 6px solid #2b6cb0; }
.section-pred  { background: linear-gradient(90deg, #e9f7ef, #f7fcf9); border-left: 6px solid #2f855a; }
.section-temp  { background: linear-gradient(90deg, #fff4e5, #fffaf3); border-left: 6px solid #dd6b20; }
.section-energy{ background: linear-gradient(90deg, #f1eaff, #faf8ff); border-left: 6px solid #805ad5; }

.pred-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin: 0.4rem 0 0.8rem 0;
}
.pred-box {
    flex: 1 1 220px;
    padding: 0.9rem 1rem;
    border-radius: 10px;
    border: 1px solid rgba(0,0,0,0.10);
    background: rgba(255,255,255,0.72);
}
.pred-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.7;
    margin-bottom: 0.2rem;
}
.pred-value { font-size: 1.25rem; font-weight: 700; }

@media (max-width: 768px) {
    .stSlider { padding: 0 0.2rem; }
    h1 { font-size: 1.5rem !important; }
    h3 { font-size: 1.1rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── header / introduction kept from the previous version ─────────────────────
st.image("Logo_long.png", width=250)
st.title("Debris Bed AI  V4.0")
st.caption("AI surrogate for ex-vessel debris bed coolability — University of Stuttgart · IKE")

with st.expander("ℹ️ Instructions & background"):
    st.markdown("""
This app predicts the coolability outcome and temperature evolution of a debris bed
formed during a severe nuclear accident, using ensemble neural network models trained
on COCOMO simulation data.

**How to use:**
1. Adjust the debris bed parameters using the sliders below.
2. The app instantly predicts whether the bed quenches or remelts.
3. If a confident prediction can be made, the predicted end time, temperature field
   evolution, and energy predictions are shown.
4. Use the slider to step manually through the temperature field frames.

*This app is for demonstration purposes and does not provide safety-relevant assessments.*
""")

st.markdown("---")

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
models      = load_models()
scaler_X    = load_scaler_X()
param_names = scaler_X.feature_names_in_.tolist()

descriptions = [
    "System pressure [MPa]",
    "Relocated fuel fraction [-]",
    "Porosity of the packed bed [-]",
    "Mean particle diameter [mm]",
    "Angle of repose [°]",
    "Flat-top radius [m]",
    "Initial bed temperature [K]",
    "Decay heat fraction [%]",
    "Structure-to-fuel mass ratio [-]",
]
min_vals = [0.11, 0.5, 0.25, 1.0, 15.0, 0.0, 400.0, 0.3, 0.25]
max_vals = [0.5,  1.0, 0.5, 5.0, 45.0, 2.0, 1700.0, 1, 2.0]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INPUT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title section-input">🎛️ 1. Debris bed parameters</div>',
            unsafe_allow_html=True)

with st.container(border=True):
    user_inputs = []
    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]
    for i in range(len(param_names)):
        with cols[i % 3]:
            val = st.slider(
                label=descriptions[i],
                min_value=float(min_vals[i]),
                max_value=float(max_vals[i]),
                value=(min_vals[i] + max_vals[i]) / 2,
                key=f"slider_{i}",
                format="%.3g"
            )
            user_inputs.append(val)

    user_inputs[0] = user_inputs[0] * 1e6      # MPa → Pa
    user_inputs[3] = user_inputs[3] / 1000.0   # mm  → m
    user_inputs[7] = user_inputs[7] / 100.0    # %   → fraction
    user_inputs_scaled = scaler_X.transform([user_inputs])[0]

# ═══════════════════════════════════════════════════════════════════════════════
# STATE + TIME CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════
Ffuel    = st.session_state["slider_1"]
Porosity = st.session_state["slider_2"]
Alpha    = st.session_state["slider_4"]
Rflat    = st.session_state["slider_5"]
Fstruct  = st.session_state["slider_8"]

rDebris, zDebris, bed_H = compute_geometry(
    MFUEL, Ffuel, Fstruct, Porosity, Rflat, Alpha)

classifier    = load_classifier()
prediction    = classifier.predict([user_inputs_scaled])[0]
probabilities = classifier.predict_proba([user_inputs_scaled])
certainty     = float(np.max(probabilities))
certainty_pct = int(certainty * 100)

avg_real             = None
predicted_duration   = None
uncertainty_duration = None

if certainty < 0.50:
    # too uncertain — no predictions at all
    avg_real = None
elif prediction == 2:
    # inconclusive — always runs to 7200 s, no regression model needed
    avg_real             = 7200.0
    predicted_duration   = timedelta(seconds=7200)
    uncertainty_duration = None   # no uncertainty — outcome is certain time limit
elif prediction in (0, 1) and models:
    cls = [0, 1] if prediction == 1 else [1, 0]
    input_array = np.concatenate(
        [np.array(user_inputs_scaled).reshape(1, -1),
         np.array(cls).reshape(1, -1)], axis=1)
    raw_preds  = [model.predict(input_array)[0][0] for model in models]
    real_preds = inverse_scaler_y(np.array(raw_preds).reshape(-1, 1)).flatten()
    avg_real             = float(np.mean(real_preds))
    std_real             = float(np.std(real_preds))
    predicted_duration   = timedelta(seconds=avg_real)
    uncertainty_duration = timedelta(seconds=std_real)

def fmt_td(td):
    s = max(0, int(td.total_seconds()))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s"

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PREDICTED OUTCOME
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title section-pred">🤖 2. Predicted outcome</div>',
            unsafe_allow_html=True)

with st.container(border=True):
    if prediction == 1:
        state_text = "♨️ Debris bed quenches"
    elif prediction == 0:
        state_text = "🌡️ Debris bed remelts"
    elif prediction == 2:
        state_text = "🤔 Inconclusive"

    if certainty < 0.50:
        time_text     = "Not available — certainty too low"
        time_unc_text = ""
    elif prediction == 2:
        time_text     = "≥ 7200 s (reaches simulation limit)"
        time_unc_text = ""
    elif predicted_duration is not None:
        time_text     = fmt_td(predicted_duration)
        time_unc_text = f"± {fmt_td(uncertainty_duration)}" if uncertainty_duration else ""
    else:
        time_text     = "Not available"
        time_unc_text = ""

    st.markdown(f"""
    <div class="pred-row">
      <div class="pred-box">
        <div class="pred-label">State prediction</div>
        <div class="pred-value">{state_text} — {certainty_pct}% certainty</div>
      </div>
      <div class="pred-box">
        <div class="pred-label">Predicted time</div>
        <div class="pred-value">{time_text} {time_unc_text}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if certainty < 0.50:
        st.warning(
            f"⚠️ State prediction certainty is only {certainty_pct}% — below 50%. "
            "Temperature and energy predictions are not available.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TEMPERATURE FIELD
# Geometry is shown ONLY when the state prediction is not sufficiently certain.
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title section-temp">🌡️ 3. Temperature field evolution</div>',
            unsafe_allow_html=True)

with st.container(border=True):
    with st.expander("ℹ️ About the temperature predictions"):
        st.markdown("""
The temperature field T(x,z,t) is predicted by a U-Net surrogate model trained on
~2,400 COCOMO simulations.

- **Left**: predicted temperature field (K)
- **Right**: ensemble uncertainty σ(K)
- Model uses the predicted end time to predict the temperature evolution from t=0 to t_end.
- Use the slider to inspect any time step manually.
""")

    if certainty < 0.50:
        st.warning(
            f"Temperature and energy predictions are withheld because the state prediction "
            f"is below 50% certainty ({certainty_pct}%).")
        st.markdown("**Debris bed geometry (shown because the state prediction is uncertain):**")
        st.pyplot(plot_geometry(rDebris, zDebris), use_container_width=True)
        st.caption(f"Bed height: {bed_H:.2f} m")
    elif not TORCH_AVAILABLE:
        st.warning("PyTorch is not available — temperature predictions are disabled.")
    elif avg_real is None:
        st.info("A reliable end-time prediction is not available for these inputs.")
    else:
        por_map_3d = build_por_map(MFUEL, Ffuel, Fstruct, Porosity, Rflat, Alpha)
        unet_models, unet_device = load_unet_ensemble()

        if unet_models is None:
            st.warning("UNet model files not found in ./unet_models/.")
        else:
            param_tuple = tuple(float(v) for v in user_inputs[:9])
            por_map_tuple = tuple(por_map_3d.ravel().tolist())

            with st.spinner("Computing temperature field — please wait..."):
                results = predict_temperature_sequence(
                    id(unet_models), param_tuple, por_map_tuple,
                    t_end_s=avg_real, n_frames=N_FRAMES)

            if results is None:
                st.error("Temperature prediction failed.")
            else:
                n_fr = len(results['times_s'])
                times = results['times_s']

                # Slider only: no Play/Pause/Reset and no playback loop.
                frame_idx = st.slider(
                    "Time step",
                    min_value=0,
                    max_value=n_fr - 1,
                    value=0,
                    key="frame_slider_main")

                st.caption(
                    f"t = {times[frame_idx]:.0f} s  "
                    f"({frame_idx + 1} / {n_fr})")

                fig = render_temperature_frame(results, frame_idx)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ENERGY PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title section-energy">⚡ 4. Energy predictions</div>',
            unsafe_allow_html=True)

with st.container(border=True):
    with st.expander("What are energy predictions?"):
        st.markdown("""
- **$E_{pd}$**: Change in total internal energy of the debris bed
- **$Q_{si}$**: Heat flux to phase change
- **$Q_{sg}$**: Heat flux to surrounding gas
- **$Q_{sl}$**: Heat flux to surrounding liquid

Error bars show ±2σ across ensemble members.
""")

    if certainty < 0.50:
        st.warning("Energy predictions are withheld because the state prediction is below 50% certainty.")
    elif predicted_duration is None:
        st.info("No reliable predicted end time is available for these inputs.")
    else:
        energy_members = load_energy_ensemble(variant="pinn")
        if not energy_members:
            st.info("No energy models found in ./energy_models/")
        else:
            try:
                # Font size for energy plots — adjust this one value
                fs = 20

                energy_times = np.linspace(0, predicted_duration.total_seconds(), 20)
                X_energy = build_energy_input(user_inputs, energy_times)

                preds_e = np.stack([
                    m.predict(X_energy, verbose=0)
                    for m in energy_members
                ])

                preds_MW = preds_e * P_0 * 1e-6
                mean_MW = preds_MW.mean(axis=0)
                std_MW = preds_MW.std(axis=0)

                # 2 × 2 layout
                fig_e, axes_e = plt.subplots(2, 2, figsize=(12, 8))
                axes_e = axes_e.flatten()

                for j, ax_e in enumerate(axes_e):
                    ax_e.errorbar(
                        energy_times,
                        mean_MW[:, j],
                        yerr=2 * std_MW[:, j],
                        fmt="o-",
                        lw=1.2,
                        markersize=4,
                        capsize=3,
                        color="#0072B2",
                        ecolor="#E69F00",
                        elinewidth=1.0
                    )

                    ax_e.set_title(
                        ENERGY_OUTPUT_LABELS[j],
                        fontsize=fs
                    )

                    ax_e.set_xlabel(
                        "Time (s)",
                        fontsize=fs
                    )

                    ax_e.tick_params(
                        labelsize=fs
                    )

                    ax_e.grid(
                        True,
                        lw=0.3,
                        alpha=0.4
                    )

                fig_e.suptitle(
                    "PINN ensemble energy predictions  |  error bars = ±2σ",
                    fontsize=fs + 1
                )

                fig_e.tight_layout()

                st.pyplot(
                    fig_e,
                    use_container_width=True
                )

                plt.close(fig_e)
            except Exception as e:
                st.error(f"Energy prediction failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.9em;'>
    <strong>Developed by:</strong> Jasmin Joshi-Thompson, University of Stuttgart,
    Institute for Nuclear Energy and Energy Systems (IKE), 2025 <br>
    <strong>Contact:</strong> Jasmin.Joshi-Thompson@ike.uni-stuttgart.de <br>
    <strong>Version:</strong> 4.0 – Last updated: September 2026<br><br>
    <em>This app uses AI models for live prediction of quench and melting times
        and energy dynamics. <strong>For demonstration purposes only<strong> and not intended for safety-relevant assessments.</em><br>
</div>
""", unsafe_allow_html=True)

# Keep the explanatory material at the bottom from the previous version.
with st.expander("Note"):
    st.markdown("""
This AI model was pre-trained with simulation data from **COCOMO** (Corium Coolability Model),
based on work published at the **NENE Conference 2026 and 2025** [1], following the work from
**NENE 2024** [2]. The simulation data was validated against experimental data from the
**FLOAT test facility** [3]. COCOMO was developed at the **Institute for Nuclear Energy and
Energy Systems (IKE)** at the **University of Stuttgart** [4].

**References:**

[1] Joshi-Thompson, J., Buck, M., and Starflinger, J., "Hybrid classification-regression neural
network for predicting coolability outcomes in ex-vessel debris bed scenarios," Proceedings of
the International Conference Nuclear Energy for New Europe, pp. 165.1–165.8, Sep. 2025.

[2] Joshi-Thompson, J., Buck, M., and Starflinger, J., "Application of AI Methods for Describing
the Coolability of Debris Beds Formed in the Late Accident Phase of Nuclear Reactors,"
Proceedings of the 33rd International Conference Nuclear Energy for New Europe (NENE 2024),
Portorož, Slovenia, September 9–12, 2024.

[3] M. Petroff, R. Kulenovic, and J. Starflinger, "Experimental investigation on debris bed
quenching with additional non-condensable gas injection," Journal of Nuclear Engineering
and Radiation Science, NERS-21-1028, 2022.

[4] Buck, M., and Pohlner, G., "Ex-Vessel Debris Bed Formation and Coolability – Challenges and
Chances for Severe Accident Mitigation," Proceedings of ICAPP 2016, San Francisco, April 2016.
""")
