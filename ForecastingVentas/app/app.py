import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forecasting Ventas · Nov 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.9rem; }
    .main-header p  { color: rgba(255,255,255,0.88); margin: 0.4rem 0 0; font-size: 1rem; }
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
        margin: 1.6rem 0;
    }
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e8e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 10px rgba(102,126,234,0.09);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_Final.joblib"
DATA_PATH  = BASE_DIR / "data" / "processed" / "inferencia_df_transformado.csv"

# ──────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES  (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


try:
    model        = load_model()
    inferencia_df = load_data()
except FileNotFoundError as exc:
    st.error(f"❌ Archivo no encontrado: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"❌ Error al cargar recursos: {exc}")
    st.stop()

FEATURE_COLS = list(model.feature_names_in_)
PRODUCTOS    = sorted(inferencia_df["nombre"].unique().tolist())

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
COMP_FACTORS = {"Actual (0%)": 1.0, "Competencia -5%": 0.95, "Competencia +5%": 1.05}


def predict_recursive(
    product_df: pd.DataFrame,
    discount_adj: float,
    comp_factor: float,
):
    """
    Predicción recursiva para los 30 días de noviembre.

    - discount_adj  : puntos porcentuales que se suman al descuento original.
    - comp_factor   : factor multiplicativo sobre precio_competencia.

    Devuelve (df_modificado, lista_predicciones).
    """
    df_p = product_df.copy().sort_values("fecha").reset_index(drop=True)

    # ── Ajustar precios ───────────────────────────────────────────────────────
    df_p["descuento_porcentaje"] = df_p["descuento_porcentaje"] + discount_adj
    df_p["precio_venta"] = (
        df_p["precio_base"] * (1.0 - df_p["descuento_porcentaje"] / 100.0)
    ).clip(lower=0.0)
    df_p["precio_competencia"] = df_p["precio_competencia"] * comp_factor
    safe_comp = df_p["precio_competencia"].replace(0, np.nan)
    df_p["ratio_precio"] = df_p["precio_venta"] / safe_comp

    # ── Añadir columnas que el modelo necesita y no están en el CSV ───────────
    for col in FEATURE_COLS:
        if col not in df_p.columns:
            if col == "black_friday" and "es_black_friday" in df_p.columns:
                df_p["black_friday"] = df_p["es_black_friday"]
            else:
                df_p[col] = 0

    # ── Historial inicial: lag_7 (más antiguo) … lag_1 (más reciente) ────────
    row0 = df_p.iloc[0]
    history = [
        float(row0["unidades_vendidas_lag_7"]),
        float(row0["unidades_vendidas_lag_6"]),
        float(row0["unidades_vendidas_lag_5"]),
        float(row0["unidades_vendidas_lag_4"]),
        float(row0["unidades_vendidas_lag_3"]),
        float(row0["unidades_vendidas_lag_2"]),
        float(row0["unidades_vendidas_lag_1"]),
    ]

    predictions = []

    for i in range(len(df_p)):
        row = df_p.iloc[[i]].copy()

        if i > 0:
            # Actualizar lags con el historial acumulado
            for lag in range(1, 8):
                row[f"unidades_vendidas_lag_{lag}"] = history[-lag]
            row["unidades_vendidas_ma7"] = float(np.mean(history[-7:]))

        X = row[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
        pred = max(0.0, float(model.predict(X)[0]))
        predictions.append(pred)
        history.append(pred)

    return df_p, predictions


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Controles de Simulación")
    st.markdown("---")

    producto_sel = st.selectbox(
        "🏷️ Producto",
        options=PRODUCTOS,
        help="Selecciona el producto que deseas simular.",
    )

    descuento_ajuste = st.slider(
        "💰 Ajuste de Descuento",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        format="%d%%",
        help=(
            "Puntos porcentuales añadidos al descuento base. "
            "Positivo → precio más bajo. Negativo → precio más alto."
        ),
    )

    escenario_comp = st.radio(
        "🏪 Escenario de Competencia",
        options=list(COMP_FACTORS.keys()),
        index=0,
        help="Variación aplicada al precio de la competencia.",
    )

    st.markdown("---")
    simular_btn = st.button(
        "🚀 Simular Ventas",
        type="primary",
        use_container_width=True,
    )
    st.markdown("---")
    st.caption("📅 Datos de **Noviembre 2025**")
    st.caption("🤖 Modelo: HistGradientBoostingRegressor")

# ──────────────────────────────────────────────────────────────────────────────
# EXECUTE PREDICTIONS
# ──────────────────────────────────────────────────────────────────────────────
if simular_btn:
    prod_df     = inferencia_df[inferencia_df["nombre"] == producto_sel].copy()
    comp_factor = COMP_FACTORS[escenario_comp]

    try:
        with st.spinner("⏳ Ejecutando predicciones recursivas día a día…"):
            df_result, main_preds = predict_recursive(
                prod_df, descuento_ajuste, comp_factor
            )

        with st.spinner("📊 Calculando comparativa de escenarios…"):
            scenario_results = {}
            for esc, cf in COMP_FACTORS.items():
                _, sc_preds = predict_recursive(prod_df, descuento_ajuste, cf)
                scenario_results[esc] = sc_preds

        st.session_state["results"] = {
            "df":           df_result,
            "predictions":  main_preds,
            "producto":     producto_sel,
            "descuento_adj": descuento_ajuste,
            "escenario":    escenario_comp,
            "scenarios":    scenario_results,
        }
    except Exception as exc:
        st.error(f"❌ Error durante la predicción: {exc}")
        st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# WELCOME SCREEN  (si aún no hay resultados)
# ──────────────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.markdown(
        """
        <div class="main-header">
            <h1>📊 Forecasting de Ventas · Noviembre 2025</h1>
            <p>Simulación de predicciones usando HistGradientBoostingRegressor</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "👈 **¡Bienvenido!** Configura los parámetros en el panel lateral "
        "y pulsa **🚀 Simular Ventas** para ver el dashboard completo."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🏷️ 24 productos")
        st.markdown("Selecciona cualquier artículo de la cartera de noviembre 2025.")
    with c2:
        st.markdown("#### 💰 Simulación de precios")
        st.markdown("Ajusta descuentos y escenarios de competencia en tiempo real.")
    with c3:
        st.markdown("#### 📈 Predicciones recursivas")
        st.markdown("Los lags se actualizan día a día para máxima precisión.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# BUILD RESULTS DATAFRAME
# ──────────────────────────────────────────────────────────────────────────────
res   = st.session_state["results"]
df_r  = res["df"].copy()
preds = res["predictions"]

df_r["unidades_predichas"]   = preds
df_r["ingresos_proyectados"] = df_r["unidades_predichas"] * df_r["precio_venta"]

precio_venta_arr = df_r["precio_venta"].values   # usado en comparativa

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="main-header">
        <h1>📊 Simulación de Ventas — Noviembre 2025</h1>
        <p>
            🏷️ <strong>{res['producto']}</strong>
            &nbsp;·&nbsp;
            💰 Descuento ajustado: <strong>{res['descuento_adj']:+d}%</strong>
            &nbsp;·&nbsp;
            🏪 <strong>{res['escenario']}</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_units    = df_r["unidades_predichas"].sum()
total_ingresos = df_r["ingresos_proyectados"].sum()
avg_precio     = df_r["precio_venta"].mean()
avg_desc       = df_r["descuento_porcentaje"].mean()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("📦 Unidades Totales", f"{total_units:,.0f}")
with k2:
    st.metric("💶 Ingresos Proyectados", f"€ {total_ingresos:,.2f}")
with k3:
    st.metric("🏷️ Precio Promedio", f"€ {avg_precio:.2f}")
with k4:
    st.metric("🎯 Descuento Promedio", f"{avg_desc:.1f}%")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── CHART: predicción diaria ──────────────────────────────────────────────────
st.subheader("📈 Predicción Diaria de Ventas — Noviembre 2025")

days     = list(range(1, len(preds) + 1))
bf_idx   = 27                          # índice 0-based de día 28
bf_day   = 28
bf_units = preds[bf_idx] if len(preds) > bf_idx else None

sns.set_theme(style="whitegrid", palette="muted")
fig, ax = plt.subplots(figsize=(13, 5))

# Línea principal
ax.plot(
    days, preds,
    color="#667eea", linewidth=2.5,
    marker="o", markersize=5,
    markerfacecolor="white", markeredgecolor="#667eea", markeredgewidth=1.5,
    zorder=3,
)
# Área bajo la curva
ax.fill_between(days, preds, alpha=0.10, color="#764ba2")

# Black Friday
if bf_units is not None:
    ax.axvline(x=bf_day, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.75, zorder=2)
    ax.scatter([bf_day], [bf_units], color="#e74c3c", s=200, zorder=5, marker="*")

    y_range  = (max(preds) - min(preds)) if max(preds) != min(preds) else 1
    offset_y = y_range * 0.18
    offset_x = -6 if bf_day > 15 else 3
    ax.annotate(
        f"Black Friday\n{bf_units:.0f} uds.",
        xy=(bf_day, bf_units),
        xytext=(bf_day + offset_x, bf_units + offset_y),
        fontsize=10,
        fontweight="bold",
        color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#e74c3c",
            alpha=0.92,
        ),
    )

ax.set_xlabel("Día de Noviembre", fontsize=12, labelpad=8)
ax.set_ylabel("Unidades Predichas", fontsize=12, labelpad=8)
ax.set_title(f"Predicción de Ventas Diaria — {res['producto']}", fontsize=14, fontweight="bold", pad=12)
ax.set_xticks(days)
ax.set_xlim(0.5, len(preds) + 0.5)
ax.tick_params(axis="x", labelsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── TABLA DETALLADA ───────────────────────────────────────────────────────────
st.subheader("📋 Tabla Detallada por Día — Noviembre 2025")

tabla = df_r[
    ["fecha", "nombre_dia", "precio_venta", "precio_competencia",
     "descuento_porcentaje", "unidades_predichas", "ingresos_proyectados"]
].copy()

tabla["Fecha"]  = tabla["fecha"].dt.strftime("%d/%m/%Y")
tabla["Día"]    = tabla["nombre_dia"].astype(str)
is_bf           = tabla["fecha"].dt.day == bf_day
tabla.loc[is_bf, "Día"] = tabla.loc[is_bf, "Día"].apply(lambda x: f"🛍️ {x} (BF)")

tabla_display = tabla[
    ["Fecha", "Día", "precio_venta", "precio_competencia",
     "descuento_porcentaje", "unidades_predichas", "ingresos_proyectados"]
].rename(columns={
    "precio_venta":         "Precio Venta (€)",
    "precio_competencia":   "Precio Competencia (€)",
    "descuento_porcentaje": "Descuento (%)",
    "unidades_predichas":   "Unidades Predichas",
    "ingresos_proyectados": "Ingresos (€)",
}).reset_index(drop=True)


def _highlight_bf(row):
    if "🛍️" in str(row["Día"]):
        return ["background-color: #fff3cd; font-weight: bold"] * len(row)
    return [""] * len(row)


styled_table = (
    tabla_display.style
    .apply(_highlight_bf, axis=1)
    .format(
        {
            "Precio Venta (€)":       "€ {:.2f}",
            "Precio Competencia (€)": "€ {:.2f}",
            "Descuento (%)":          "{:.1f}%",
            "Unidades Predichas":     "{:.0f}",
            "Ingresos (€)":           "€ {:,.2f}",
        },
        na_rep="—",
    )
)

st.dataframe(styled_table, use_container_width=True, height=530)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── COMPARATIVA DE ESCENARIOS ─────────────────────────────────────────────────
st.subheader("🔄 Comparativa de Escenarios de Competencia")
st.caption(
    f"Descuento aplicado en todos los escenarios: **{res['descuento_adj']:+d}%**. "
    "Solo varía el precio de la competencia."
)

SCENARIO_META = {
    "Actual (0%)":     ("🟡", "Sin cambios en la competencia."),
    "Competencia -5%": ("🔴", "Competencia baja precios 5% → nuestra posición empeora."),
    "Competencia +5%": ("🟢", "Competencia sube precios 5% → nuestra posición mejora."),
}

base_preds    = np.array(res["scenarios"]["Actual (0%)"])
base_units    = float(base_preds.sum())
base_ingresos = float(np.dot(base_preds, precio_venta_arr))

sc1, sc2, sc3 = st.columns(3)
cols_map = {"Actual (0%)": sc1, "Competencia -5%": sc2, "Competencia +5%": sc3}

for esc, col_widget in cols_map.items():
    sc_arr      = np.array(res["scenarios"][esc])
    sc_units    = float(sc_arr.sum())
    sc_ingresos = float(np.dot(sc_arr, precio_venta_arr))
    icon, desc  = SCENARIO_META[esc]

    delta_u  = f"{sc_units - base_units:+.0f}"    if esc != "Actual (0%)" else None
    delta_i  = f"€ {sc_ingresos - base_ingresos:+,.2f}" if esc != "Actual (0%)" else None

    with col_widget:
        st.markdown(f"**{icon} {esc}**")
        st.caption(desc)
        st.metric("Unidades Totales",       f"{sc_units:,.0f}",        delta=delta_u)
        st.metric("Ingresos Proyectados",   f"€ {sc_ingresos:,.2f}",   delta=delta_i)

