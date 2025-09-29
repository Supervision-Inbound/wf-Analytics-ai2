#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

# =========================
# CONFIG AJUSTABLE
# =========================
CLIMA_URL = os.getenv(
    "CLIMA_URL",
    "https://turnos-api-inbound.andres-eliecergonzalez.workers.dev/clima"
)

# Donde escribir las alertas
OUT_PATH = "public/alertas.json"

# Archivo con el baseline horario
SEMILLA_PATH = "data/semilla_llamadas.json"

# Mapeo opcional de nombres de comuna (para corregir tildes/variantes)
COMUNAS_MAPPING_PATH = "data/comunas_mapping.json"  # opcional

# Sensibilidad por comuna (opcional). Si no tienes, se usa sensibilidad media (1.0)
SENSIBILIDAD_PATH = "data/sensibilidad_comunas.json"

# Parámetros de severidad y ponderación
RAIN_MM_MIN = 5.0     # desde este umbral empieza a “pegar” la lluvia
RAIN_MM_MAX = 30.0    # a partir de aquí consideramos severidad lluvia≈1
WIND_KMH_MIN = 35.0   # desde aquí viento empieza a “pegar”
WIND_KMH_MAX = 60.0   # a partir de aquí severidad viento≈1

K_BASE = 0.25         # impacto máximo (25%) cuando severidad=1 y sensibilidad=1

# Filtro para no alertar ruido
MIN_EXTRA_ABS = 5                  # mínimo de extra_calls en números absolutos
MIN_EXTRA_REL = 0.08               # o 8% del baseline, lo que sea mayor
CONFIDENCE_MIN = 0.35              # descartar alertas de muy baja confianza

TZ = "America/Santiago"            # para ventana futura 7 días
# =========================


# -------------------------
# Utilidades
# -------------------------
def load_json_if_exists(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default if default is not None else {}
    except Exception:
        return default if default is not None else {}


def normalize_name(s: str) -> str:
    """Normaliza nombre de comuna: minúsculas, sin dobles espacios, recorta."""
    if not s:
        return s
    s2 = str(s).strip().lower()
    s2 = " ".join(s2.split())
    # puedes meter normalizaciones rápidas aquí si quieres
    return s2


def now_scl():
    # Runner está en UTC; calculamos ventana desde SCL por simplicidad (aprox)
    # 03:00 SCL ~ 06:00 UTC; aquí solo nos interesa "hoy-->+7 días"
    return dt.datetime.utcnow() - dt.timedelta(hours=3)


def combine_severity(rain_sev: float, wind_sev: float) -> float:
    """
    Combina dos severidades (0..1) sin doble contabilidad.
    Usamos 1 - (1-rain)*(1-wind) ≈ unión probabilística.
    """
    rain_sev = max(0.0, min(1.0, rain_sev))
    wind_sev = max(0.0, min(1.0, wind_sev))
    return 1.0 - (1.0 - rain_sev) * (1.0 - wind_sev)


def severity_rain(precip_mm: float, precip_prob: float) -> float:
    """
    Severidad lluvia continua con probabilidad:
    - mm por encima de RAIN_MM_MIN sube lineal hasta RAIN_MM_MAX
    - multiplicamos por probabilidad (0..1)
    """
    if precip_mm is None:
        return 0.0
    mm = max(0.0, precip_mm - RAIN_MM_MIN) / max(1e-6, (RAIN_MM_MAX - RAIN_MM_MIN))
    mm = max(0.0, min(1.0, mm))
    p = max(0.0, min(1.0, (precip_prob or 0.0) / 100.0))
    return mm * p


def severity_wind(wind_kmh: float) -> float:
    """Severidad viento continua (0..1) entre WIND_KMH_MIN y WIND_KMH_MAX."""
    if wind_kmh is None:
        return 0.0
    sev = (wind_kmh - WIND_KMH_MIN) / max(1e-6, (WIND_KMH_MAX - WIND_KMH_MIN))
    return max(0.0, min(1.0, sev))


def confidence_from(rain_sev: float, wind_sev: float, precip_prob: float) -> float:
    """
    Confianza simple: mezcla de severidades y probabilidad de precipitación.
    """
    p = max(0.0, min(1.0, (precip_prob or 0.0)/100.0))
    sev = combine_severity(rain_sev, wind_sev)
    return 0.5 * sev + 0.5 * p


def expected_from_seed(seed: dict, when: dt.datetime) -> float:
    """
    Calcula expected_calls desde la semilla.
    Admite:
      - {"hourly": [24 valores], "weekend_boost": 1.10, "weekday_boost": 1.0}
      - {"weekday": {0..6: [24 valores]}}
      - {"by_hour": [24]} (fallback)
    """
    h = when.hour
    dow = when.weekday()

    if "weekday" in seed and isinstance(seed["weekday"], dict):
        row = seed["weekday"].get(str(dow)) or seed["weekday"].get(dow)
        if isinstance(row, (list, tuple)) and len(row) == 24:
            return float(row[h])

    if "hourly" in seed and isinstance(seed["hourly"], list) and len(seed["hourly"]) == 24:
        base = float(seed["hourly"][h])
        # boosts
        if dow >= 5:
            base *= float(seed.get("weekend_boost", 1.0))
        else:
            base *= float(seed.get("weekday_boost", 1.0))
        return base

    if "by_hour" in seed and isinstance(seed["by_hour"], list) and len(seed["by_hour"]) == 24:
        return float(seed["by_hour"][h])

    # fallback muy básico
    return 100.0


def build_equivalences(mapping: dict) -> dict:
    """
    Construye un diccionario de equivalencias normalizadas.
    mapping puede traer { "pto montt": "puerto montt", ... }.
    """
    eq = {}
    for k, v in (mapping or {}).items():
        eq[normalize_name(k)] = normalize_name(v)
    return eq


def main():
    # ---------- Carga insumos ----------
    seed = load_json_if_exists(SEMILLA_PATH, default={"hourly": [100]*24})
    sensibilidad = load_json_if_exists(SENSIBILIDAD_PATH, default={})  # {comuna_norm: 0.6..1.4}
    mapping_raw = load_json_if_exists(COMUNAS_MAPPING_PATH, default={})
    mapping = build_equivalences(mapping_raw)

    # Sensibilidad default (media)
    def sens_for(comuna_norm: str) -> float:
        return float(sensibilidad.get(comuna_norm, 1.0))

    # ---------- Descarga clima ----------
    r = requests.get(CLIMA_URL, timeout=60)
    r.raise_for_status()
    clima = r.json()

    # Estructura esperada (por comuna):
    # {
    #   "comuna": "Puerto Montt",
    #   "lat": -41.45,
    #   "lon": -72.93,
    #   "hourly": { "time": [...], "wind_speed_10m": [...], "precipitation": [...], "precipitation_probability": [...] }
    # }
    df_all = []
    for loc in clima:
        comuna_src = loc.get("comuna") or loc.get("name")
        if not comuna_src:
            continue
        comuna_norm = mapping.get(normalize_name(comuna_src), normalize_name(comuna_src))

        h = loc.get("hourly", {})
        times = h.get("time") or []
        ws = h.get("wind_speed_10m") or []
        pr_mm = h.get("precipitation") or []
        pr_p = h.get("precipitation_probability") or []

        n = min(len(times), len(ws), len(pr_mm), len(pr_p))
        for i in range(n):
            df_all.append({
                "comuna_raw": comuna_src,
                "comuna": comuna_norm,
                "time_utc": times[i],
                "wind_speed_10m": ws[i],
                "precip_mm": pr_mm[i],
                "precip_prob": pr_p[i],
            })

    if not df_all:
        raise RuntimeError("No llegaron datos de clima o estructura inesperada.")

    df = pd.DataFrame(df_all)
    # parse fechas (son locales en tu worker → asumiremos ISO local o UTC; si vienen ISO-Z, pandas las capta)
    df["dt"] = pd.to_datetime(df["time_utc"], errors="coerce")
    df = df.dropna(subset=["dt"])

    # Ventana: hoy (SCL) → +7 días
    t0 = now_scl().replace(minute=0, second=0, microsecond=0)
    tmax = t0 + dt.timedelta(days=7)
    df = df[(df["dt"] >= t0) & (df["dt"] <= tmax)].copy()

    # Cálculo de severidades y expected
    df["rain_sev"] = df.apply(lambda r: severity_rain(r["precip_mm"], r["precip_prob"]), axis=1)
    df["wind_sev"] = df["wind_speed_10m"].apply(severity_wind)
    df["sev"] = df.apply(lambda r: combine_severity(r["rain_sev"], r["wind_sev"]), axis=1)

    # expected_calls desde semilla por hora/dow
    df["expected_calls"] = df["dt"].apply(lambda d: expected_from_seed(seed, d))

    # sensibilidad comuna
    df["sens"] = df["comuna"].apply(sens_for)

    # extra_calls continuo
    df["extra_calls_cont"] = df["expected_calls"] * K_BASE * df["sens"] * df["sev"]

    # Redondeo y filtros
    df["extra_calls"] = np.rint(df["extra_calls_cont"]).astype(int)
    df["total_calls"] = (df["expected_calls"] + df["extra_calls"]).astype(int)

    # confidence
    df["confidence"] = df.apply(lambda r: confidence_from(r["rain_sev"], r["wind_sev"], r["precip_prob"]), axis=1)

    # Tipo de evento “más dominante”
    def classify_evt(row):
        if row["rain_sev"] >= 0.6 and row["wind_sev"] >= 0.6:
            return "lluvia+viento"
        if row["rain_sev"] >= row["wind_sev"]:
            return "lluvia" if row["rain_sev"] > 0 else "ninguno"
        return "viento" if row["wind_sev"] > 0 else "ninguno"

    df["evento"] = df.apply(classify_evt, axis=1)

    # Filtro de ruido
    def pass_threshold(row):
        min_rel = max(MIN_EXTRA_ABS, int(math.ceil(MIN_EXTRA_REL * row["expected_calls"])))
        return (row["extra_calls"] >= min_rel) and (row["confidence"] >= CONFIDENCE_MIN)

    df = df[df.apply(pass_threshold, axis=1)].copy()

    # Ordenar y seleccionar columnas finales
    df = df.sort_values(["dt", "comuna"])
    out = []
    for _, r in df.iterrows():
        out.append({
            "fecha": r["dt"].strftime("%Y-%m-%d"),
            "hora": int(r["dt"].hour),
            "comuna": r["comuna"],
            "evento": r["evento"],
            "wind_speed_10m": float(round(r["wind_speed_10m"], 2)),
            "precip_mm": float(round(r["precip_mm"], 2)),
            "precip_prob": int(round(r["precip_prob"])),
            "expected_calls": int(round(r["expected_calls"])),
            "extra_calls": int(round(r["extra_calls"])),
            "total_calls": int(round(r["total_calls"])),
            "severity": float(round(r["sev"], 3)),
            "confidence": float(round(r["confidence"], 3))
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ alertas guardadas en {OUT_PATH} — filas: {len(out)}")


if __name__ == "__main__":
    main()

