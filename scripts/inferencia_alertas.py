# scripts/inferencia_alertas.py
# Genera public/alertas.json usando:
#  - Sensibilidad por comuna (models/sensibilidad_comunas.json) aprendida en Kaggle
#  - Clima futuro desde tu Worker (hoy -> +7 días)
#  - Baseline esperado por hora (data/semilla_llamadas.json)
#
# Salida por alerta:
#   fecha_dt, hora, comuna, evento, lluvia_mm, viento_km_h,
#   llamadas_base, llamadas_adicionales, llamadas_esperadas, motivo

import os, json, math, re, unicodedata
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import numpy as np
import requests

# ========= CONFIG =========
CLIMA_URL = "https://turnos-api-inbound.andres-eliecergonzalez.workers.dev/clima"

SENS_PATH   = "models/sensibilidad_comunas.json"
SEED_PATH   = "data/semilla_llamadas.json"
OUT_PATH    = "public/alertas.json"

# Zona horaria (preferimos zoneinfo si está disponible)
try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("America/Santiago")
except Exception:
    TZ = timezone(timedelta(hours=-3))  # fallback

# Horizonte: HOY → +7 días
HORIZON_DAYS  = 7
HORIZON_HOURS = HORIZON_DAYS * 24

# Umbrales de evento
HEAVY_RAIN_MM   = 5.0
STRONG_WIND_KMH = 35.0

# Para publicar alerta se exige:
MIN_EXTRA_CALLS      = 15.0        # llamadas adicionales mínimas
REQUIERE_EVENTO_METEO = True       # solo alertamos si hay lluvia/viento fuerte

# Mapeos manuales (si detectas diferencias de nombre entre API y entrenamiento)
MANUAL_MAP: Dict[str, str] = {
    # "Pto Montt": "Puerto Montt",
}

# ========= Utils =========
def normalize_name(s: str) -> str:
    """Normaliza nombres de comuna para matcheo robusto."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def now_scl() -> datetime:
    return datetime.now(tz=TZ)

def clamp(x: float, lo: float = 0.0) -> float:
    return float(x) if x >= lo else 0.0

# ========= Cargar sensibilidad del entrenamiento =========
if not os.path.exists(SENS_PATH):
    raise FileNotFoundError(f"No se encontró {SENS_PATH}. Sube models/sensibilidad_comunas.json desde Kaggle.")

with open(SENS_PATH, "r", encoding="utf-8") as f:
    S = json.load(f)

keep_cols     = S["keep_cols"]
mu            = np.array(S["scaler_mean"],  dtype=float)
sigma         = np.array(S["scaler_scale"], dtype=float)
beta          = np.array(S["ridge_coef"],   dtype=float)
idx_by_comuna = {k: list(v) for k, v in S["idx_by_comuna"].items()}  # {comuna: [idxs]}

sigma_safe = np.where(sigma == 0.0, 1.0, sigma)
col_index = {c: i for i, c in enumerate(keep_cols)}

trained_norm_map = {normalize_name(c): c for c in idx_by_comuna.keys()}

def map_comuna(api_name: str) -> Optional[str]:
    if api_name in MANUAL_MAP:
        return MANUAL_MAP[api_name]
    return trained_norm_map.get(normalize_name(api_name))

# ========= Baseline por hora (semilla) =========
def load_seed(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        # seed mínimo para no romper (media 100)
        return {"global_mean": 100.0, "baseline_by_dow_hour": {}}
    with open(path, "r", encoding="utf-8") as f:
        seed = json.load(f)
    return seed

SEED = load_seed(SEED_PATH)

def baseline_for(dt: datetime) -> float:
    """
    Devuelve baseline esperado para la hora 'dt' (Santiago).
    Acepta varios formatos de semilla:
      - {"baseline_by_dow_hour": {"0-13": v, ...}, "global_mean": v}
      - {"avg_by_dow_hour": {...}}
      - {"by_hour_of_week": [168 valores] (0=lunes 00:00)}
      - {"global_mean": v}
    """
    dow = dt.weekday()  # 0=lunes .. 6=domingo
    hr  = dt.hour
    key = f"{dow}-{hr}"
    # 1) clave explícita dow-hora
    for kname in ("baseline_by_dow_hour", "avg_by_dow_hour"):
        m = SEED.get(kname, {})
        if isinstance(m, dict) and key in m:
            try:
                return float(m[key])
            except Exception:
                pass
    # 2) vector por hora de semana
    hov = SEED.get("by_hour_of_week")
    if isinstance(hov, (list, tuple)) and len(hov) >= 168:
        idx = dow * 24 + hr
        try:
            return float(hov[idx])
        except Exception:
            pass
    # 3) global
    gm = SEED.get("global_mean")
    try:
        return float(gm)
    except Exception:
        return 100.0

# ========= Obtener clima futuro =========
resp = requests.get(CLIMA_URL, timeout=60)
resp.raise_for_status()
clima = resp.json()
if not clima.get("ok"):
    raise RuntimeError("El endpoint de clima no devolvió ok=true")
rows = clima.get("rows", [])
if not rows:
    raise RuntimeError("El endpoint de clima no trae filas (rows vacío).")

# ========= Generación de alertas =========
ahora  = now_scl()
corte  = ahora + timedelta(hours=HORIZON_HOURS)

base_x = mu.copy()

def set_val(x: np.ndarray, col: str, val: float):
    j = col_index.get(col)
    if j is not None:
        x[j] = float(val)

alertas = []

for loc in rows:
    api_name = loc.get("name", "")
    comuna_train = map_comuna(api_name)
    if not comuna_train:
        # comuna no vista en entrenamiento → ignoramos
        continue

    idxs = idx_by_comuna.get(comuna_train, [])
    if not idxs:
        continue

    h = (loc.get("hourly") or {})
    times = h.get("time") or []
    wind  = h.get("wind_speed_10m") or []
    rain  = h.get("precipitation") or []

    n = min(len(times), len(wind), len(rain))
    for i in range(n):
        # timestamp → SCL
        try:
            t_utc = datetime.fromisoformat(times[i].replace("Z", "+00:00"))
            t = t_utc.astimezone(TZ)
        except Exception:
            continue

        if t < ahora or t > corte:
            continue

        viento_kmh = float(wind[i]) if wind[i] is not None else 0.0
        lluvia_mm  = float(rain[i])  if rain[i]  is not None else 0.0

        # Evento meteo (para filtrar ruido si se desea)
        lluvia_fuerte = lluvia_mm  >= HEAVY_RAIN_MM
        viento_fuerte = viento_kmh >= STRONG_WIND_KMH
        if REQUIERE_EVENTO_METEO and not (lluvia_fuerte or viento_fuerte):
            continue

        # Construir vector x en el mismo orden que keep_cols
        x = base_x.copy()
        set_val(x, f"{comuna_train}__Lluvia_mm",     lluvia_mm)
        set_val(x, f"{comuna_train}__Viento_km_h",   viento_kmh)
        set_val(x, f"{comuna_train}__lluvia_fuerte", 1 if lluvia_fuerte else 0)
        set_val(x, f"{comuna_train}__viento_fuerte", 1 if viento_fuerte else 0)

        x_s = (x - mu) / sigma_safe
        delta = float(np.sum(x_s[idxs] * beta[idxs]))  # llamadas adicionales por evento en esa comuna/hora

        if delta < MIN_EXTRA_CALLS:
            continue

        # Baseline esperado para esa fecha/hora
        base_calls = max(0.0, baseline_for(t))
        total_calls = max(0.0, base_calls + delta)

        if lluvia_fuerte and viento_fuerte:
            evento = "mixto"
        elif lluvia_fuerte:
            evento = "lluvia"
        elif viento_fuerte:
            evento = "viento"
        else:
            evento = "suave"

        alertas.append({
            "fecha_dt": t.strftime("%Y-%m-%d"),
            "hora": t.hour,
            "comuna": comuna_train,
            "evento": evento,
            "lluvia_mm": round(lluvia_mm, 2),
            "viento_km_h": round(viento_kmh, 1),
            "llamadas_base": round(base_calls, 1),
            "llamadas_adicionales": round(delta, 1),
            "llamadas_esperadas": round(total_calls, 1),
            "motivo": "modelo_sensibilidad_comuna"
        })

# Ordenar y guardar
alertas.sort(key=lambda r: (r["fecha_dt"], r["hora"], r["comuna"]))
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(alertas, f, ensure_ascii=False, indent=2)

print(f"✅ Generadas {len(alertas)} alertas (hoy→+{HORIZON_DAYS} días) → {OUT_PATH}")
