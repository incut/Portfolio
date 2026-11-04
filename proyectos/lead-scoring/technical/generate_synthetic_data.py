"""Generate the 21-column schema in memory without reading source records."""
import numpy as np
import pandas as pd


def generate(rows=600, seed=42):
    if rows < 100:
        raise ValueError("Use at least 100 synthetic rows for nested cross-validation")
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "id": [f"synthetic-{number:06d}" for number in range(rows)],
        "origen": rng.choice(["API", "Landing Page Submission"], rows),
        "fuente": rng.choice(["Google", "Direct Traffic", "Organic Search"], rows),
        "no_enviar_email": rng.choice(["Yes", "No"], rows),
        "no_llamar": rng.choice(["Yes", "No"], rows),
        "visitas_total": rng.poisson(3, rows).astype(float),
        "tiempo_en_site_total": rng.gamma(2, 150, rows),
        "paginas_vistas_visita": rng.uniform(0, 5, rows),
        "ult_actividad": rng.choice(["SMS Sent", "Email Opened", "Page Visited on Website"], rows),
        "ambito": rng.choice(["Select", "Business Administration"], rows),
        "ocupacion": rng.choice(["Student", "Working Professional", "Unemployed"], rows),
        "score_actividad": rng.uniform(0, 100, rows),
        "score_perfil": rng.uniform(0, 100, rows),
        "descarga_lm": rng.choice(["Yes", "No"], rows),
    })
    for channel in ("google", "revista", "periodico", "youtube", "facebook", "referencias"):
        frame[f"conociste_{channel}"] = rng.choice(["Yes", "No"], rows)
    probability = np.where(frame.ocupacion.eq("Working Professional"), 0.65, 0.25)
    frame["compra"] = rng.binomial(1, probability)
    missing = rng.choice(rows, rows // 20, replace=False)
    frame.loc[missing, ["score_actividad", "score_perfil"]] = np.nan
    return frame
