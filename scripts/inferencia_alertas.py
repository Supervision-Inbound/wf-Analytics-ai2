import json
import math
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from unidecode import unidecode

# =========================
# Entradas por variables de entorno
# =========================
CLIMA_URL   = os.getenv("CLIMA_URL", "").strip()
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "public/alertas.json")
SEED_PATH   = os.getenv("SEED_PATH", "data/semilla_llamadas.json")

# Horizonte máximo (días) hacia adelante para exportar alertas
MAX_DAYS_AHEAD = 7

SCL = ZoneInfo("America/Santiago")

# =========================
# Utilidades
# =========================
def norm(txt: str) -> str:
    """Normaliza nombres de comuna para aumentar el match."""
    if not isinstance(txt, str):
        return ""
    t = unidecode(txt).lower().strip()
    t = (
        t.replace("pto.", "puerto")
         .replace("pto ", "puerto ")
         .replace("  ", " ")
    )
    return t

def _seed_to_dict(raw) -> dict:
    """
    Convierte distintos formatos de semilla a:
    {
      'comunas': [{'name':'...', '_norm':'...'}, ...],
      'umbral_lluvia_mm': float|None,
      'umbral_viento_kmh': float|None,
      'umbral_prob_lluvia_pct': float|None
    }
    """
    out = {
        "comunas": [],
        "umbral_lluvia_mm": None,
        "umbral_viento_kmh": None,
        "umbral_prob_lluvia_pct": None,
    }

    # Caso 1: objeto con comunas + umbrales
    if isinstance(raw, dict):
        # comunas puede venir como lista de strings u objetos
        comunas = raw.get("comunas", [])
        if isinstance(comunas, list):
            for c in comunas:
                if isinstance(c, str):
                    out["comunas"].append({"name": c})
                elif isinstance(c, dict):
                    name = c.get("name") or c.get("comuna") or ""
                    if name:
                        out["comunas"].append({"name": name})
        # umbrales si existen
        for k in ["umbral_lluvia_mm", "umbral_viento_kmh", "umbral_prob_lluvia_pct"]:
            if k in raw:
                try:
                    out[k] = float(raw[k])
                except Exception:
                    pass
        return out

    # Caso 2: lista
    if isinstance(raw, list):
        # puede ser lista de strings o de objetos con 'name'
        for c in raw:
            if isinstance(c, str):
                out["comunas"].append({"name": c})
            elif isinstance(c, dict):
                name = c.get("name") or c.get("comuna") or ""
                if name:
                    out["comunas"].append({"name": name})
        return out

    # desconocido
    return out

def load_seed(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    seed = _seed_to_dict(raw)

    # Normaliza nombres
    for c in seed.get("comunas", []):
        c["_norm"] = norm(c.get("name", ""))

    # Umbrales por defecto si faltan
    if seed.get("umbral_lluvia_mm") is None:
        seed["umbral_lluvia_mm"] = 5.0
    if seed.get("umbral_viento_kmh") is None:
        seed["umbral_viento_kmh"] = 35.0
    if seed.get("umbral_prob_lluvia_pct") is None:
        seed["umbral_prob_lluvia_pct"] = 60.0

    return seed

def fetch_clima(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def iter_hours_from_row(row: dict):
    """Itera horas del registro climático del Worker (/clima)."""
    h = row.get("hourly", {}) or {}
    times = h.get("time", []) or []
    wind  = h.get("wind_speed_10m", []) or []
    rain  = h.get("precipitation", []) or []
    ppop  = h.get("precipitation_probability", []) or []
    n = min(len(times), len(wind), len(rain), len(ppop))
    for i in range(n):
        yield times[i], float(wind[i]), float(rain[i]), float(ppop[i])

def parse_iso_local_to_scl(ts: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts.replace("Z",""))
    except Exception:
        dt = datetime.strptime(ts[:16], "%Y-%m-%dT%H:%M")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=SCL)
    else:
        dt = dt.astimezone(SCL)
    return dt

# =========================
# Inferencia simple de alertas
# =========================
def main():
    if not CLIMA_URL:
        raise SystemExit("CLIMA_URL no definido")

    seed = load_seed(SEED_PATH)
    data = fetch_clima(CLIMA_URL)

    U_LLUVIA = float(seed.get("umbral_lluvia_mm", 5.0))
    U_VIENTO = float(seed.get("umbral_viento_kmh", 35.0))
    U_PPOP   = float(seed.get("umbral_prob_lluvia_pct", 60.0))

    comunas_seed = seed.get("comunas", [])
    comunas_seed_map = {c["_norm"]: c.get("name") for c in comunas_seed}

    alerts = []
    now_scl = datetime.now(SCL)
    limit_dt = now_scl + timedelta(days=MAX_DAYS_AHEAD)

    rows = data.get("rows", []) if isinstance(data, dict) else []
    for r in rows:
        comuna_name = r.get("name", "")
        comuna_norm = norm(comuna_name)

        for ts, wind, rain, ppop in iter_hours_from_row(r):
            dt = parse_iso_local_to_scl(ts)

            # Solo desde AHORA en Santiago
            if dt < now_scl:
                continue
            if dt > limit_dt:
                continue

            lluvia_fuerte = rain >= U_LLUVIA
            viento_fuerte = wind >= U_VIENTO
            prob_lluvia   = ppop >= U_PPOP

            if not (lluvia_fuerte or viento_fuerte or prob_lluvia):
                continue

            extra = 0
            if lluvia_fuerte:
                extra += 10 + 2 * max(0, rain - U_LLUVIA)
            if viento_fuerte:
                extra += 6 + 0.5 * max(0, wind - U_VIENTO)
            if prob_lluvia and not lluvia_fuerte:
                extra += 4

            alerts.append({
                "fecha": dt.date().isoformat(),
                "hora": dt.hour,
                "comuna": comunas_seed_map.get(comuna_norm, comuna_name),
                "clima": {
                    "lluvia_mm": round(rain, 2),
                    "viento_km_h": round(wind, 1),
                    "prob_lluvia_pct": round(ppop, 0)
                },
                "motivo": {
                    "lluvia_fuerte": lluvia_fuerte,
                    "viento_fuerte": viento_fuerte,
                    "prob_lluvia_alta": prob_lluvia
                },
                "llamadas_extra_est": int(max(0, round(extra)))
            })

    alerts.sort(key=lambda a: (a["fecha"], a["hora"], a["comuna"]))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2)

    print(f"✅ alertas generadas: {len(alerts)}")
    if alerts:
        print(f"Rango: {alerts[0]['fecha']} {alerts[0]['hora']:02d}:00 → {alerts[-1]['fecha']} {alerts[-1]['hora']:02d}:00")


if __name__ == "__main__":
    main()
