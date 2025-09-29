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

# === Zona horaria (robusto) ===
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TZ_SCL = ZoneInfo("America/Santiago")
except Exception:
    TZ_SCL = None  # fallback sin zoneinfo

# =========================
# CONFIG AJUSTABLE (por env o editando aquí)
# =========================
CLIMA_URL = os.getenv(
    "CLIMA_URL",
    "https://turnos-api-inbound.andres-eliecergonzalez.workers.dev/clima"
)

# Salida
OUT_PATH = "public/alertas.json"

# Insumos locales
SEMILLA_PATH            = "data/semilla_llamadas.json"     # baseline por (dow-hora)
COMUNAS_MAPPING_PATH    = "data/comunas_mapping.json"       # opcional (sinónimo → nombre oficial)
SENSIBILIDAD_PATH       = "data/sensibilidad_comunas.json"  # opcional (ponderador por comuna)

# Parámetros de severidad y ponderación (realismo)
RAIN_MM_MIN   = float(os.getenv("RAIN_MM_MIN", "5.0"))   # mm a partir de donde "pega"
RAIN_MM_MAX   = float(os.getenv("RAIN_MM_MAX", "30.0"))  # mm donde severidad ≈ 1
WIND_KMH_MIN  = float(os.getenv("WIND_KMH_MIN", "35.0")) # km/h a partir de donde "pega"
WIND_KMH_MAX  = float(os.getenv("WIND_KMH_MAX", "60.0")) # km/h donde severidad ≈ 1

# Intensidad base del efecto meteo (fracción máx. sobre expected_calls)
# factor_model = 1 + K_BASE * severidad (antes de sensibilidad/alpha)
K_BASE = float(os.getenv("K_BASE", "0.30"))               # 30% de techo con severidad=1

# Sensibilidad y filtros de alertas
ALPHA_SUAVIZADO  = float(os.getenv("ALPHA_SUAVIZADO", "0.75"))
UMBRAL_FACTOR    = float(os.getenv("UMBRAL_FACTOR", "1.10"))  # 10%+
UMBRAL_MIN_CALLS = int(os.getenv("UMBRAL_MIN_CALLS", "8"))    # mín. llamadas extra
CONFIDENCE_MIN   = float(os.getenv("CONFIDENCE_MIN", "0.35")) # descartar baja confianza

# =========================
# Utils
# =========================
def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default if default is not None else {}
    except Exception:
        return default if default is not None else {}

def normalize_name(s: str) -> str:
    if not s:
        return s
    s2 = str(s).strip().lower()
    s2 = " ".join(s2.split())
    return s2

def build_equivalences(mapping: dict) -> dict:
    eq = {}
    for k, v in (mapping or {}).items():
        eq[normalize_name(k)] = normalize_name(v)
    return eq

def now_scl():
    # Si hay zoneinfo disponible, usamos SCL; si no, UTC “aproximado”
    if TZ_SCL is not None:
        return dt.datetime.now(tz=TZ_SCL)
    return dt.datetime.utcnow()

# ---------- Baseline (semilla) ----------
def load_seed():
    """
    Acepta estructuras:
      1) {"baseline_by_dow_hour": {"0-0": x, "0-1": y, ...}, "global_mean": m}
      2) {"weekday": {"0":[24],..., "6":[24]}, "global_mean": m}
      3) {"hourly":[24], "weekday_boost":1.0, "weekend_boost":1.05}
      4) Fallback: global_mean=100
    """
    seed = load_json(SEMILLA_PATH, default={})
    # Normaliza a dict "baseline_by_dow_hour"
    if "baseline_by_dow_hour" in seed and isinstance(seed["baseline_by_dow_hour"], dict):
        base = {str(k): float(v) for k, v in seed["baseline_by_dow_hour"].items()}
        gm = float(seed.get("global_mean", 100.0))
        return base, gm

    if "weekday" in seed and isinstance(seed["weekday"], dict):
        base = {}
        for k, arr in seed["weekday"].items():
            if isinstance(arr, list) and len(arr) == 24:
                for h in range(24):
                    base[f"{k}-{h}"] = float(arr[h])
        gm = float(seed.get("global_mean", np.mean(list(base.values())) if base else 100.0))
        return base, gm

    if "hourly" in seed and isinstance(seed["hourly"], list) and len(seed["hourly"]) == 24:
        wb = float(seed.get("weekday_boost", 1.0))
        we = float(seed.get("weekend_boost", 1.0))
        base = {}
        for dow in range(7):
            boost = we if dow >= 5 else wb
            for h in range(24):
                base[f"{dow}-{h}"] = float(seed["hourly"][h]) * boost
        gm = float(seed.get("global_mean", np.mean(seed["hourly"])))
        return base, gm

    # Fallback llano
    base = {f"{dow}-{h}": 100.0 for dow in range(7) for h in range(24)}
    return base, 100.0

def expected_from_seed(baseline_by_dow_hour: dict, global_mean: float, when: dt.datetime) -> float:
    # when está en tz SCL; usamos weekday() y hour
    key = f"{when.weekday()}-{when.hour}"
    return float(baseline_by_dow_hour.get(key, global_mean))

# ---------- Severidades y confianza ----------
def severity_rain(precip_mm: float, precip_prob: float) -> float:
    if precip_mm is None:
        return 0.0
    mm = max(0.0, (precip_mm - RAIN_MM_MIN)) / max(1e-6, (RAIN_MM_MAX - RAIN_MM_MIN))
    mm = max(0.0, min(1.0, mm))
    p = max(0.0, min(1.0, (precip_prob or 0.0) / 100.0))
    return mm * p

def severity_wind(wind_kmh: float) -> float:
    if wind_kmh is None:
        return 0.0
    sev = (wind_kmh - WIND_KMH_MIN) / max(1e-6, (WIND_KMH_MAX - WIND_KMH_MIN))
    return max(0.0, min(1.0, sev))

def combine_severity(rain_sev: float, wind_sev: float) -> float:
    return 1.0 - (1.0 - max(0, min(1, rain_sev))) * (1.0 - max(0, min(1, wind_sev)))

def confidence_from(rain_sev: float, wind_sev: float, precip_prob: float) -> float:
    p = max(0.0, min(1.0, (precip_prob or 0.0) / 100.0))
    sev = combine_severity(rain_sev, wind_sev)
    return 0.5 * sev + 0.5 * p

# ---------- Sensibilidad ----------
def load_sensitivity():
    sens = load_json(SENSIBILIDAD_PATH, default={})
    return {normalize_name(k): float(v) for k, v in sens.items()}

def adjust_factor(factor_model: float, comuna_norm: str, sens_map: dict) -> float:
    override = float(sens_map.get(comuna_norm, 1.0))
    return (factor_model ** ALPHA_SUAVIZADO) * override

# ---------- Fetch y parsing de clima ----------
def fetch_clima(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Soportar dos formas:
    # a) { ok: true, rows: [ {...}, {...} ] }
    # b) [ {...}, {...} ]
    if isinstance(data, dict) and "rows" in data:
        rows = data["rows"]
    elif isinstance(data, list):
        rows = data
    else:
        raise RuntimeError("Estructura de clima no reconocida.")

    out = []
    for loc in rows:
        comuna_src = loc.get("comuna") or loc.get("name")
        h = loc.get("hourly") or {}
        times = h.get("time") or []
        ws = h.get("wind_speed_10m") or []
        pr_mm = h.get("precipitation") or []
        pr_p = h.get("precipitation_probability") or []
        n = min(len(times), len(ws), len(pr_mm), len(pr_p))
        out.append({
            "comuna": comuna_src,
            "time": times[:n],
            "wind": ws[:n],
            "rain_mm": pr_mm[:n],
            "rain_prob": pr_p[:n],
        })
    return out

# =========================
# Main
# =========================
def main():
    # ----- Cargar insumos -----
    baseline_by_dow_hour, global_mean = load_seed()
    mapping_raw = load_json(COMUNAS_MAPPING_PATH, default={})
    mapping = build_equivalences(mapping_raw)
    sens_map = load_sensitivity()

    # ----- Descargar clima -----
    clima_rows = fetch_clima(CLIMA_URL)

    # ----- Expandir a DataFrame -----
    recs = []
    for loc in clima_rows:
        comuna_raw = loc.get("comuna")
        if not comuna_raw:
            continue
        comuna_norm = normalize_name(comuna_raw)
        comuna_norm = mapping.get(comuna_norm, comuna_norm)

        times = loc["time"]
        ws = loc["wind"]
        rm = loc["rain_mm"]
        rp = loc["rain_prob"]

        for i in range(len(times)):
            recs.append({
                "comuna_raw": comuna_raw,
                "comuna_norm": comuna_norm,
                "time": times[i],
                "wind_kmh": ws[i],
                "precip_mm": rm[i],
                "precip_prob": rp[i],
            })

    if not recs:
        raise RuntimeError("Clima vacío o estructura inesperada.")

    df = pd.DataFrame(recs)

    # ======= PARSE DATETIME ROBUSTO (arreglo) =======
    # 1) parsear siempre en UTC (naive → UTC; aware → convertido a UTC)
    ts_utc = pd.to_datetime(df["time"], errors="coerce", utc=True)
    # 2) convertir a America/Santiago si es posible
    if TZ_SCL is not None:
        df["dt"] = ts_utc.dt.tz_convert(TZ_SCL)
    else:
        # fallback sin zoneinfo (mantener en UTC)
        df["dt"] = ts_utc
    # ================================================

    df = df.dropna(subset=["dt"])

    # Ventana: hoy(SCL) → +7 días (incluye hora actual redondeada)
    now = now_scl().replace(minute=0, second=0, microsecond=0)
    tmax = now + dt.timedelta(days=7)
    df = df[(df["dt"] >= now) & (df["dt"] <= tmax)].copy()

    # Severidades y confianza
    df["rain_sev"] = df.apply(lambda r: severity_rain(r["precip_mm"], r["precip_prob"]), axis=1)
    df["wind_sev"] = df["wind_kmh"].apply(severity_wind)
    df["sev"] = df.apply(lambda r: combine_severity(r["rain_sev"], r["wind_sev"]), axis=1)
    df["confidence"] = df.apply(lambda r: confidence_from(r["rain_sev"], r["wind_sev"], r["precip_prob"]), axis=1)

    # Baseline esperado por hora (según dow-hora SCL)
    df["expected_calls"] = df["dt"].apply(lambda d: expected_from_seed(baseline_by_dow_hour, global_mean, d))

    # Factor por clima (antes de sensibilidad): 1 + K_BASE * severidad
    df["factor_model"] = 1.0 + (K_BASE * df["sev"]).clip(lower=0.0)

    # Ajuste por sensibilidad por comuna + suavizado
    df["factor_adj"] = df.apply(lambda r: adjust_factor(r["factor_model"], r["comuna_norm"], sens_map), axis=1)

    # Adicionales y totales
    df["extra_calls"] = np.rint(df["expected_calls"] * np.maximum(0.0, df["factor_adj"] - 1.0)).astype(int)
    df["total_calls"] = (df["expected_calls"] + df["extra_calls"]).astype(int)

    # Etiqueta evento dominante
    def classify_evt(row):
        if row["rain_sev"] >= 0.6 and row["wind_sev"] >= 0.6:
            return "lluvia+viento"
        if row["rain_sev"] >= row["wind_sev"]:
            return "lluvia" if row["rain_sev"] > 0 else "ninguno"
        return "viento" if row["wind_sev"] > 0 else "ninguno"

    df["evento"] = df.apply(classify_evt, axis=1)

    # Filtros: ruido y confianza
    def pass_threshold(row):
        if row["confidence"] < CONFIDENCE_MIN:
            return False
        if row["factor_adj"] < UMBRAL_FACTOR:
            return False
        # requerir también un mínimo absoluto relativo a la base
        min_abs = max(UMBRAL_MIN_CALLS, int(math.ceil(0.08 * row["expected_calls"])))
        return row["extra_calls"] >= min_abs

    df = df[df.apply(pass_threshold, axis=1)].copy()

    # Ordenar y preparar salida
    df = df.sort_values(["dt", "comuna_norm"])
    out = []
    for _, r in df.iterrows():
        out.append({
            "fecha": r["dt"].strftime("%Y-%m-%d"),
            "hora": int(r["dt"].hour),
            "comuna": r["comuna_norm"],
            "evento": r["evento"],
            "wind_speed_10m": float(round(r["wind_kmh"], 2)),
            "precip_mm": float(round(r["precip_mm"], 2)),
            "precip_prob": int(round(r["precip_prob"])),
            "expected_calls": int(round(r["expected_calls"])),
            "extra_calls": int(round(r["extra_calls"])),
            "total_calls": int(round(r["total_calls"])),
            "severity": float(round(r["sev"], 3)),
            "confidence": float(round(r["confidence"], 3)),
            "factor_model": float(round(r["factor_model"], 3)),
            "factor_ajustado": float(round(r["factor_adj"], 3))
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ alertas guardadas en {OUT_PATH} — filas: {len(out)}")

if __name__ == "__main__":
    main()
