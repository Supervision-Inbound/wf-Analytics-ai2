#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math
import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

# ====== CONFIG (con valores estrictos por defecto) ======
CLIMA_URL = os.getenv("CLIMA_URL", "https://turnos-api-inbound.andres-eliecergonzalez.workers.dev/clima")

# Salida
OUT_PATH = "public/alertas.json"

# Insumos locales
SEMILLA_PATH         = "data/semilla_llamadas.json"         # baseline esperada (dow-hora)
COMUNAS_MAPPING_PATH = "data/comunas_mapping.json"          # opcional { "pto montt": "puerto montt", ... }
SENSIBILIDAD_PATH    = "data/sensibilidad_comunas.json"     # opcional { "puerto montt": 1.2, ... }

# Umbrales de evento/meteo (modo estricto)
RAIN_MM_MIN   = float(os.getenv("RAIN_MM_MIN",   "3.0"))   # mm mínimos para “lluvia”
RAIN_MM_MAX   = float(os.getenv("RAIN_MM_MAX",   "30.0"))  # mm donde severidad≈1
WIND_KMH_MIN  = float(os.getenv("WIND_KMH_MIN",  "45.0"))  # viento relevante
WIND_KMH_MAX  = float(os.getenv("WIND_KMH_MAX",  "60.0"))  # viento muy fuerte

# Intensidad base del efecto meteo (techo porcentual sobre baseline)
K_BASE = float(os.getenv("K_BASE", "0.25"))                 # 25% con severidad=1

# Sensibilidad y filtros de publicación (más realistas)
ALPHA_SUAVIZADO  = float(os.getenv("ALPHA_SUAVIZADO", "0.70"))
UMBRAL_FACTOR    = float(os.getenv("UMBRAL_FACTOR", "1.15"))    # ≥ +15%
UMBRAL_MIN_CALLS = int(os.getenv("UMBRAL_MIN_CALLS", "15"))     # ≥ +15 llamadas
CONFIDENCE_MIN   = float(os.getenv("CONFIDENCE_MIN", "0.55"))   # confianza mínima

# ====== Zona horaria SCL (exacta con zoneinfo) ======
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TZ_SCL = ZoneInfo("America/Santiago")
except Exception:
    # Fallback simple (si faltara base IANA): usa aproximación -03
    TZ_SCL = None


# ---------- Helpers ----------
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
    if TZ_SCL is not None:
        return dt.datetime.now(TZ_SCL)
    # Fallback a -03 si no hay zoneinfo
    return dt.datetime.utcnow() - dt.timedelta(hours=3)

# ---------- Baseline (semilla) ----------
def load_seed():
    """
    Formatos soportados:
      1) {"baseline_by_dow_hour": {"0-0": x, ...}, "global_mean": m}
      2) {"weekday": {"0":[24],..., "6":[24]}, "global_mean": m}
      3) {"hourly":[24], "weekday_boost":1.0, "weekend_boost":1.05}
      4) Fallback: global_mean=100
    """
    seed = load_json(SEMILLA_PATH, default={})
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

    base = {f"{dow}-{h}": 100.0 for dow in range(7) for h in range(24)}
    return base, 100.0

def expected_from_seed(baseline_by_dow_hour: dict, global_mean: float, when: dt.datetime) -> float:
    key = f"{when.weekday()}-{when.hour}"
    return float(baseline_by_dow_hour.get(key, global_mean))

# ---------- Severidades / confianza ----------
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

# ---------- Sensibilidad por comuna ----------
def load_sensitivity():
    sens = load_json(SENSIBILIDAD_PATH, default={})
    return {normalize_name(k): float(v) for k, v in sens.items()}

def adjust_factor(factor_model: float, comuna_norm: str, sens_map: dict) -> float:
    override = float(sens_map.get(comuna_norm, 1.0))
    return (factor_model ** ALPHA_SUAVIZADO) * override

# ---------- Clima ----------
def fetch_clima(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
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
        ws    = h.get("wind_speed_10m") or []
        pr_mm = h.get("precipitation") or []
        pr_p  = h.get("precipitation_probability") or []
        n = min(len(times), len(ws), len(pr_mm), len(pr_p))
        out.append({
            "comuna": comuna_src,
            "time": times[:n],
            "wind": ws[:n],
            "rain_mm": pr_mm[:n],
            "rain_prob": pr_p[:n],
        })
    return out

# ---------- Main ----------
def main():
    # Insumos
    baseline_by_dow_hour, global_mean = load_seed()
    mapping = build_equivalences(load_json(COMUNAS_MAPPING_PATH, default={}))
    sens_map = load_sensitivity()

    # Clima → registros fila-a-fila
    clima_rows = fetch_clima(CLIMA_URL)
    recs = []
    for loc in clima_rows:
        comuna_raw = loc.get("comuna")
        if not comuna_raw:
            continue
        comuna_norm = normalize_name(comuna_raw)
        comuna_norm = mapping.get(comuna_norm, comuna_norm)

        for t, w, rmm, rp in zip(loc["time"], loc["wind"], loc["rain_mm"], loc["rain_prob"]):
            recs.append({
                "comuna_raw":  comuna_raw,
                "comuna_norm": comuna_norm,
                "time": t,
                "wind_kmh": w,
                "precip_mm": rmm,
                "precip_prob": rp,
            })
    if not recs:
        raise RuntimeError("Clima vacío o estructura inesperada.")

    df = pd.DataFrame(recs)
    # Timestamps a tz SCL y recorte HOY→+7d
    if TZ_SCL is not None:
        df["dt"] = pd.to_datetime(df["time"], errors="coerce").dt.tz_convert(TZ_SCL)
        # si la fuente viene naive, localiza
        if df["dt"].isna().any():
            df["dt"] = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize(TZ_SCL, nonexistent="shift_forward", ambiguous="NaT")
    else:
        df["dt"] = pd.to_datetime(df["time"], errors="coerce")  # fallback
    df = df.dropna(subset=["dt"])

    today_scl = now_scl().replace(hour=0, minute=0, second=0, microsecond=0)
    limit_scl = today_scl + dt.timedelta(days=7)
    df = df[(df["dt"] >= today_scl) & (df["dt"] <= limit_scl)].copy()

    # Severidades / confianza
    df["rain_sev"] = df.apply(lambda r: severity_rain(r["precip_mm"], r["precip_prob"]), axis=1)
    df["wind_sev"] = df["wind_kmh"].apply(severity_wind)
    df["sev"]      = df.apply(lambda r: combine_severity(r["rain_sev"], r["wind_sev"]), axis=1)
    df["confidence"] = df.apply(lambda r: confidence_from(r["rain_sev"], r["wind_sev"], r["precip_prob"]), axis=1)

    # Baseline esperado
    def _expected(d):
        # si tiene tz, úsala; si no, toma naive
        if hasattr(d, "tzinfo") and d.tzinfo is not None:
            d_local = d
        else:
            d_local = d
        return expected_from_seed(baseline_by_dow_hour, global_mean, d_local)

    df["expected_calls"] = df["dt"].apply(_expected)

    # Factor por clima (antes de sensibilidad)
    df["factor_model"] = 1.0 + (K_BASE * df["sev"]).clip(lower=0.0)

    # Ajuste por sensibilidad
    df["factor_adj"] = df.apply(lambda r: adjust_factor(r["factor_model"], r["comuna_norm"], sens_map), axis=1)

    # Impacto
    df["extra_calls"] = np.rint(df["expected_calls"] * np.maximum(0.0, df["factor_adj"] - 1.0)).astype(int)
    df["total_calls"] = (df["expected_calls"] + df["extra_calls"]).astype(int)

    # Evento (estricto)
    def classify_evt(row):
        wind_evt = row["wind_kmh"] >= max(WIND_KMH_MIN, 45.0)
        rain_evt = (row["precip_mm"] >= max(RAIN_MM_MIN, 3.0)) and (row["precip_prob"] >= 70)
        if wind_evt and rain_evt:
            return "lluvia+viento"
        if rain_evt:
            return "lluvia"
        if wind_evt:
            return "viento"
        return "ninguno"

    df["evento"] = df.apply(classify_evt, axis=1)

    # Filtros de publicación
    def pass_threshold(row):
        if row["evento"] == "ninguno":
            return False
        if row["sev"] < 0.60:
            return False
        if row["confidence"] < max(CONFIDENCE_MIN, 0.55):
            return False
        if row["factor_adj"] < max(UMBRAL_FACTOR, 1.15):
            return False
        min_abs = max(UMBRAL_MIN_CALLS, int(math.ceil(0.08 * row["expected_calls"])))
        if row["extra_calls"] < min_abs:
            return False
        return True

    df = df[df.apply(pass_threshold, axis=1)].copy()

    # Orden y salida
    df = df.sort_values(["dt", "comuna_norm"])
    out = []
    for _, r in df.iterrows():
        # Fecha/hora en SCL humano
        dt_local = r["dt"]
        if hasattr(dt_local, "tzinfo") and dt_local.tzinfo is not None:
            fecha = dt_local.strftime("%Y-%m-%d")
            hora  = int(dt_local.hour)
        else:
            fecha = pd.to_datetime(dt_local).strftime("%Y-%m-%d")
            hora  = int(pd.to_datetime(dt_local).hour)

        out.append({
            "fecha": fecha,
            "hora": hora,
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
