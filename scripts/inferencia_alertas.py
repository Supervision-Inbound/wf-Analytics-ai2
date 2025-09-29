import json
import math
import os
from datetime import datetime, date, time, timedelta
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

# =========================
# Utilidades
# =========================
SCL = ZoneInfo("America/Santiago")

def norm(txt: str) -> str:
    """Normaliza nombres de comuna para aumentar el match."""
    if not isinstance(txt, str):
        return ""
    t = unidecode(txt).lower().strip()
    # normalizaciones manuales comunes
    t = (
        t.replace("pto.", "puerto")
         .replace("pto ", "puerto ")
         .replace("  ", " ")
    )
    return t

def load_seed(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        seed = json.load(f)
    # seed: { "comunas": [{"name": "...", ...}, ...], "umbral_lluvia_mm": 5, ... }
    # Forzamos normalización de nombres semilla
    for c in seed.get("comunas", []):
        c["_norm"] = norm(c.get("name", ""))
    return seed

def fetch_clima(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def iter_hours_from_row(row: dict):
    """Itera horas del registro climático del Worker (/clima).
       row['hourly'] contiene arrays paralelos: time[], precipitation[], precipitation_probability[], wind_speed_10m[]
    """
    h = row.get("hourly", {}) or {}
    times = h.get("time", []) or []
    wind  = h.get("wind_speed_10m", []) or []
    rain  = h.get("precipitation", []) or []
    ppop  = h.get("precipitation_probability", []) or []
    n = min(len(times), len(wind), len(rain), len(ppop))
    for i in range(n):
        yield times[i], float(wind[i]), float(rain[i]), float(ppop[i])

def parse_iso_local_to_scl(ts: str) -> datetime:
    # El Worker retorna times locales (timezone=auto). Parseamos y “decimos” que ya están en SCL.
    # Si vinieran con Z/offset, se podría usar fromisoformat y astimezone(SCL).
    try:
        # Maneja "YYYY-MM-DDTHH:MM" o con segundos
        dt = datetime.fromisoformat(ts.replace("Z",""))
    except Exception:
        # fallback muy defensivo
        dt = datetime.strptime(ts[:16], "%Y-%m-%dT%H:%M")
    # asumimos hora local de Chile
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=SCL)
    else:
        dt = dt.astimezone(SCL)
    return dt

# =========================
# Inferencia muy simple de alertas
#   - Usa umbrales de la semilla (lluvia/viento/prob lluvia)
#   - Estima llamadas extra como factor simple (ejemplo)
# =========================
def main():
    if not CLIMA_URL:
        raise SystemExit("CLIMA_URL no definido")

    seed = load_seed(SEED_PATH)
    data = fetch_clima(CLIMA_URL)

    # Umbrales desde semilla (con valores por defecto razonables)
    U_LLUVIA = float(seed.get("umbral_lluvia_mm", 5.0))
    U_VIENTO = float(seed.get("umbral_viento_kmh", 35.0))
    U_PPOP   = float(seed.get("umbral_prob_lluvia_pct", 60.0))

    # Comunas “conocidas” por el modelo/semilla (para matchear rápido)
    comunas_seed = seed.get("comunas", [])
    comunas_seed_map = {c["_norm"]: c.get("name") for c in comunas_seed}

    alerts = []
    now_scl = datetime.now(SCL)
    limit_dt = now_scl + timedelta(days=MAX_DAYS_AHEAD)

    # El Worker devuelve un objeto con 'rows': lista de comunas agregadas por país.
    rows = data.get("rows", []) if isinstance(data, dict) else []
    for r in rows:
        comuna_name = r.get("name", "")
        comuna_norm = norm(comuna_name)

        # Si no está en la semilla, igual podemos considerarla; opcionalmente la ignoras:
        # if comuna_norm not in comunas_seed_map: continue

        for ts, wind, rain, ppop in iter_hours_from_row(r):
            dt = parse_iso_local_to_scl(ts)

            # ====== FILTRO FECHA/HORA: solo desde AHORA en Santiago ======
            if dt < now_scl:
                continue
            # ====== Limitar horizonte ======
            if dt > limit_dt:
                continue

            # Señales de mal clima
            lluvia_fuerte = rain >= U_LLUVIA
            viento_fuerte = wind >= U_VIENTO
            prob_lluvia   = ppop >= U_PPOP

            if not (lluvia_fuerte or viento_fuerte or prob_lluvia):
                continue

            # Estimación “dummy” de llamadas extra (ajusta si quieres)
            # Podrías usar factores de semilla por comuna o algo más afinado.
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
                "comuna": comuna_seed_map.get(comuna_norm, comuna_name),  # nombre “bonito” si está en semilla
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

    # Ordenar y volcar
    alerts.sort(key=lambda a: (a["fecha"], a["hora"], a["comuna"]))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2)

    print(f"✅ alertas generadas: {len(alerts)}")
    if alerts:
        print(f"Rango: {alerts[0]['fecha']} {alerts[0]['hora']:02d}:00 → {alerts[-1]['fecha']} {alerts[-1]['hora']:02d}:00")


if __name__ == "__main__":
    main()
