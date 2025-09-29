# scripts/inferencia_alertas.py
import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Any
from unidecode import unidecode

CLIMA_URL   = os.environ.get("CLIMA_URL", "https://turnos-api-inbound.andres-eliecergonzalez.workers.dev/clima")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "public/alertas.json")
SEED_PATH   = os.environ.get("SEED_PATH", "data/semilla_llamadas.json")

HEAVY_RAIN_MM        = 5.0
MODERATE_RAIN_MM     = 2.0
LIGHT_RAIN_MM        = 1.0
STRONG_WIND_KMH      = 35.0
MODERATE_WIND_KMH    = 25.0
LIGHT_WIND_KMH       = 15.0
PR_RAIN_SEVERE       = 70
PR_RAIN_MOD          = 50
PR_RAIN_LIGHT        = 30

FACTOR_LIGHT   = 1.05
FACTOR_MOD     = 1.15
FACTOR_SEVERE  = 1.35

def _norm(s: str) -> str:
    if s is None:
        return ""
    t = unidecode(s).lower()
    for a, b in {
        "’": "'", "´": "'", "`": "'", "‘": "'", ".": " ", ",": " ", ";": " ", ":": " ",
        "(": " ", ")": " ", "/": " ", "-": " ", "_": " ", "”": '"', "“": '"'
    }.items():
        t = t.replace(a, b)
    t = t.replace("pto ", "puerto ").replace("pto. ", "puerto ")
    return " ".join(t.split())

ALIAS = {
    "pto montt": "puerto montt",
    "pto. montt": "puerto montt",
    "p montt": "puerto montt",
    "o higgin": "o'higgins",
    "o'higgin": "o'higgins",
    "o higgins": "o'higgins",
}

def build_canonical_map_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping = {}
    for r in rows:
        raw = r.get("name", "")
        n = _norm(raw)
        if n and n not in mapping:
            mapping[n] = raw
    for alias, target_norm in ALIAS.items():
        if target_norm in mapping:
            mapping[alias] = mapping[target_norm]
    return mapping

def canonicalize(name: str, canon_map: Dict[str, str]) -> str:
    n = _norm(name)
    return canon_map.get(n, name)

def classify_severity(rain_mm: float, wind_kmh: float, p_rain: float):
    reasons, level = [], "none"
    if rain_mm >= HEAVY_RAIN_MM or p_rain >= PR_RAIN_SEVERE:
        reasons.append("lluvia_fuerte"); level = "severe"
    elif rain_mm >= MODERATE_RAIN_MM or p_rain >= PR_RAIN_MOD:
        reasons.append("lluvia_moderada"); level = "moderate"
    elif rain_mm >= LIGHT_RAIN_MM or p_rain >= PR_RAIN_LIGHT:
        reasons.append("lluvia_leve"); level = "light"
    if wind_kmh >= STRONG_WIND_KMH:
        reasons.append("viento_fuerte"); level = "severe"
    elif wind_kmh >= MODERATE_WIND_KMH and level != "severe":
        reasons.append("viento_moderado"); level = "moderate"
    elif wind_kmh >= LIGHT_WIND_KMH and level not in ("severe", "moderate"):
        reasons.append("viento_leve"); level = "light"
    return level, reasons

def factor_by_level(level: str) -> float:
    return {"severe": 1.35, "moderate": 1.15, "light": 1.05}.get(level, 1.0)

def load_seed(path: str) -> Dict[str, int]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = {}
    for row in data:
        date = row.get("fecha_dt")
        hour = int(row.get("hora", 0))
        pred = int(max(0, row.get("pred_llamadas", 0)))
        key = f"{date}T{hour:02d}:00"
        m[key] = pred
    return m

def main():
    resp = requests.get(CLIMA_URL, timeout=60)
    resp.raise_for_status()
    clima = resp.json()
    rows = clima.get("rows", [])

    canon_map = build_canonical_map_from_rows(rows)
    seed = load_seed(SEED_PATH)

    out = []
    for comuna in rows:
        raw_name = comuna.get("name", "")
        name = canonicalize(raw_name, canon_map)
        lat  = float(comuna.get("lat", 0))
        lon  = float(comuna.get("lon", 0))

        hourly = comuna.get("hourly", {})
        times  = hourly.get("time", [])
        wind   = hourly.get("wind_speed_10m", [])
        rain   = hourly.get("precipitation", [])
        prain  = hourly.get("precipitation_probability", [])

        n = min(len(times), len(wind), len(rain), len(prain))
        for i in range(n):
            ts = str(times[i])
            try:
                dt = datetime.fromisoformat(ts.replace("Z", ""))
                key = f"{dt.strftime('%Y-%m-%d')}T{dt.hour:02d}:00"
            except Exception:
                key = ts[:13] + ":00"

            level, reasons = classify_severity(float(rain[i]), float(wind[i]), float(prain[i]))
            if level == "none":
                continue

            base = seed.get(key, 0)
            factor = factor_by_level(level)
            pred_calls = int(round(base * factor))
            uplift = max(0, pred_calls - base)

            out.append({
                "datetime_local": key,
                "comuna": name,
                "lat": lat,
                "lon": lon,
                "level": level,
                "reasons": reasons,
                "baseline_calls": base,
                "pred_calls": pred_calls,
                "expected_additional_calls": uplift
            })

    out.sort(key=lambda r: (r["datetime_local"], r["comuna"]))
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ Generado {OUTPUT_PATH} con {len(out)} alertas")

if __name__ == "__main__":
    main()
