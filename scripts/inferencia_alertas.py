# scripts/inferencia_alertas.py
# Requiere: pandas, numpy, lightgbm, joblib, requests, pytz
import os, json, math, warnings
from datetime import datetime, timedelta, timezone
import pytz
import requests
import numpy as np
import pandas as pd
from joblib import load

# ================== CONFIG ==================
CLIMA_FUTURO_URL = os.getenv("CLIMA_FUTURO_URL", "https://<tu-worker>/clima")  # <-- PÓN tu URL real
OUT_ALERTAS_PATH = "public/alertas.json"
TZ = "America/Santiago"
HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", "7"))

# Umbrales para generar alerta
MIN_REL_INCREASE = float(os.getenv("MIN_REL_INCREASE", "0.10"))  # +10% mínimo
MIN_ABS_INCREASE = int(os.getenv("MIN_ABS_INCREASE", "10"))      # +10 llamadas

# Rutas artefactos
MODEL_PATH = "models/modelo_trafico_clima_lgbm_llamadas.pkl"
FEATURES_PATH = "models/features.json"  # opcional
SEED_PATH = "data/semilla_llamadas.json"  # opcional (últimas 24-48h)
# ============================================

DEFAULT_FEATURES = [
    "hour_sin","hour_cos","dow","month","is_weekend",
    "Temp_prom_nac","Lluvia_total_nac","Viento_prom_nac",
    "lluvia_sum_6h","viento_max_6h","temp_mean_6h",
    "llamadas_lag1","llamadas_lag24","llamadas_rolling_mean_3h"
]

def load_json_url(url: str):
    r = requests.get(url, headers={"accept":"application/json"}, timeout=60)
    r.raise_for_status()
    return r.json()

def parse_clima(raw):
    """
    Espera el formato de tu Worker /clima:
    {
      rows: [
        { name, timezone, hourly:{ time[], precipitation[], precipitation_probability[], wind_speed_10m[] } }
      ]
    }
    Devuelve lista de dicts por comuna con arrays alineados.
    """
    if isinstance(raw, dict) and "rows" in raw and isinstance(raw["rows"], list):
        out = []
        for r in raw["rows"]:
            h = r.get("hourly", {})
            out.append({
                "comuna": r.get("name"),
                "timezone": r.get("timezone", "auto"),
                "time": h.get("time", []),
                "precip": h.get("precipitation", []),
                "prob": h.get("precipitation_probability", []),
                "wind": h.get("wind_speed_10m", []),
            })
        return out
    raise ValueError("Formato de clima desconocido para este script.")

def to_local_date_hour(iso_str, tz=TZ):
    d = datetime.fromisoformat(str(iso_str).replace("Z","+00:00"))
    local = d.astimezone(pytz.timezone(tz))
    return local.strftime("%Y-%m-%d"), local.hour

def build_national_timeseries(clima_series):
    """
    Agrega por hora a nivel nacional:
      - Temp_prom_nac (proxy con 0; tu forecast no trae temperatura)
      - Lluvia_total_nac  (sum mm)
      - Viento_prom_nac   (mean km/h)
      - Ventanas 6h: lluvia_sum_6h, viento_max_6h, temp_mean_6h
    """
    rows = {}
    for serie in clima_series:
        n = min(len(serie["time"]), len(serie["precip"]), len(serie["wind"]))
        for i in range(n):
            fecha, hora = to_local_date_hour(serie["time"][i], TZ)
            key = f"{fecha}T{hora:02d}"
            precip = float(serie["precip"][i] or 0.0)
            wind   = float(serie["wind"][i] or 0.0)
            if key not in rows:
                rows[key] = {"fecha_dt":fecha, "hora":hora, "Lluvia_total_nac":0.0, "viento_vals":[]}
            rows[key]["Lluvia_total_nac"] += precip
            rows[key]["viento_vals"].append(wind)

    # promedios de viento + columnas
    recs=[]
    for key, v in rows.items():
        import numpy as _np
        Vprom = _np.mean(v["viento_vals"]) if v["viento_vals"] else 0.0
        recs.append({
            "fecha_dt": v["fecha_dt"],
            "hora": int(v["hora"]),
            "Temp_prom_nac": 0.0,
            "Lluvia_total_nac": float(v["Lluvia_total_nac"]),
            "Viento_prom_nac": float(Vprom)
        })
    df = pd.DataFrame(recs)
    if df.empty:
        raise RuntimeError("Clima nacional vacío")

    # idx datetime para rolling 6h
    dt = pd.to_datetime(df["fecha_dt"]) + pd.to_timedelta(df["hora"], unit="h")
    df = df.set_index(dt).sort_index()

    # rolling 6h
    df["lluvia_sum_6h"] = df["Lluvia_total_nac"].rolling("6h", min_periods=1).sum()
    df["viento_max_6h"] = df["Viento_prom_nac"].rolling("6h", min_periods=1).max()
    df["temp_mean_6h"]  = df["Temp_prom_nac"].rolling("6h", min_periods=1).mean()

    df = df.reset_index(drop=True)
    return df

def add_calendar_features(df):
    df["fecha_dt"] = pd.to_datetime(df["fecha_dt"])
    df["dow"] = df["fecha_dt"].dt.dayofweek
    df["month"] = df["fecha_dt"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df["hora"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hora"]/24)
    return df

def load_features():
    if os.path.exists(FEATURES_PATH):
        try:
            with open(FEATURES_PATH, "r") as f:
                js = json.load(f)
                feats = js.get("features") or js
                if isinstance(feats, list) and feats:
                    return feats
        except Exception:
            pass
    return [
        "hour_sin","hour_cos","dow","month","is_weekend",
        "Temp_prom_nac","Lluvia_total_nac","Viento_prom_nac",
        "lluvia_sum_6h","viento_max_6h","temp_mean_6h",
        "llamadas_lag1","llamadas_lag24","llamadas_rolling_mean_3h"
    ]

def load_seed_series():
    if not os.path.exists(SEED_PATH):
        return None
    try:
        with open(SEED_PATH, "r") as f:
            arr = json.load(f)
        df = pd.DataFrame(arr)
        df["fecha_dt"] = pd.to_datetime(df["fecha_dt"])
        df = df.sort_values(["fecha_dt","hora"])
        return df
    except Exception:
        return None

def weekday_hour_baseline(df_seed, target_col="llamadas"):
    if df_seed is None or df_seed.empty or target_col not in df_seed.columns:
        return {(wd,h):0.0 for wd in range(7) for h in range(24)}
    tmp = df_seed.copy()
    tmp["wd"] = tmp["fecha_dt"].dt.dayofweek
    mp = {}
    for wd in range(7):
        for h in range(24):
            vals = tmp.loc[(tmp["wd"]==wd) & (tmp["hora"]==h), target_col].values
            mp[(wd,h)] = float(np.mean(vals)) if len(vals) else 0.0
    return mp

def factor_clima(precip_mm, prob_pct, wind_kmh):
    f = 1.0
    if precip_mm > 10: f *= 1.20
    elif precip_mm >= 5: f *= 1.10
    elif precip_mm >= 1: f *= 1.05
    if prob_pct >= 70: f *= 1.05
    if wind_kmh >= 35: f *= 1.05
    return f

def etiqueta_clima(precip_mm, prob_pct, wind_kmh):
    tags = []
    if precip_mm > 10: tags.append("lluvia_fuerte")
    elif precip_mm >= 5: tags.append("lluvia_moderada")
    elif precip_mm >= 1: tags.append("lluvia_leve")
    if prob_pct >= 70: tags.append("alta_prob_precip")
    if wind_kmh >= 35: tags.append("viento_fuerte")
    return ",".join(tags) if tags else "sin_evento"

def main():
    os.makedirs("public", exist_ok=True)

    # 1) Cargar modelo + features
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    model = load(MODEL_PATH)
    features = load_features()

    # 2) Cargar clima futuro y parsear
    raw = load_json_url(CLIMA_FUTURO_URL)
    clima_series = parse_clima(raw)

    # recortar horizonte a N días
    now = datetime.now(timezone.utc)
    max_ts = now + timedelta(days=HORIZON_DAYS)
    for s in clima_series:
        keep = []
        for i, tiso in enumerate(s["time"]):
            try:
                ts = datetime.fromisoformat(str(tiso).replace("Z","+00:00"))
            except Exception:
                continue
            if ts <= max_ts:
                keep.append(i)
        for k in ("time","precip","prob","wind"):
            s[k] = [s[k][i] for i in keep]

    # 3) Clima nacional por hora + calendario
    df_nat = build_national_timeseries(clima_series)
    df_nat = add_calendar_features(df_nat)

    # 4) Lags (semilla opcional)
    seed = load_seed_series()
    wd_hour_avg = weekday_hour_baseline(seed, "llamadas")

    df_nat = df_nat.sort_values(["fecha_dt","hora"]).reset_index(drop=True)
    df_nat["pred_base"] = 0.0
    last_by_key = {}
    if seed is not None and not seed.empty:
        for _, r in seed.iterrows():
            key = f"{r['fecha_dt'].strftime('%Y-%m-%d')}T{int(r['hora']):02d}"
            last_by_key[key] = float(r["llamadas"])

    rolling_buff = []

    # 5) Predicción secuencial (usa lags a medida que avanza)
    for i, row in df_nat.iterrows():
        fecha = pd.to_datetime(row["fecha_dt"]).strftime("%Y-%m-%d")
        hora  = int(row["hora"])
        key   = f"{fecha}T{hora:02d}"

        dt_curr = pd.to_datetime(row["fecha_dt"])
        wd = dt_curr.dayofweek

        prev_dt = dt_curr + pd.to_timedelta(hora-1, unit="h")
        prev_key = (prev_dt if (hora-1)>=0 else dt_curr - pd.Timedelta(days=1) + pd.to_timedelta(23, unit="h"))
        prev_key = f"{prev_key.strftime('%Y-%m-%d')}T{((hora-1)%24):02d}"
        lag1 = last_by_key.get(prev_key, wd_hour_avg.get((wd, (hora-1)%24), 0.0))

        lag24_dt = dt_curr - pd.Timedelta(days=1)
        lag24_key = f"{lag24_dt.strftime('%Y-%m-%d')}T{hora:02d}"
        lag24 = last_by_key.get(lag24_key, wd_hour_avg.get(((wd-1)%7, hora), 0.0))

        if rolling_buff:
            rm3 = float(np.mean(rolling_buff[-3:]))
        else:
            rm3 = wd_hour_avg.get((wd, hora), 0.0)

        row_feats = {
            "hour_sin": row["hour_sin"],
            "hour_cos": row["hour_cos"],
            "dow": int(row["dow"]),
            "month": int(row["month"]),
            "is_weekend": int(row["is_weekend"]),
            "Temp_prom_nac": float(row["Temp_prom_nac"]),
            "Lluvia_total_nac": float(row["Lluvia_total_nac"]),
            "Viento_prom_nac": float(row["Viento_prom_nac"]),
            "lluvia_sum_6h": float(row["lluvia_sum_6h"]),
            "viento_max_6h": float(row["viento_max_6h"]),
            "temp_mean_6h": float(row["temp_mean_6h"]),
            "llamadas_lag1": float(lag1),
            "llamadas_lag24": float(lag24),
            "llamadas_rolling_mean_3h": float(rm3),
        }
        x = np.array([[row_feats.get(f, 0.0) for f in features]], dtype=float)
        pred = float(model.predict(x)[0])
        pred = max(0.0, pred)
        df_nat.at[i, "pred_base"] = pred

        last_by_key[key] = pred
        rolling_buff.append(pred)

    # 6) Alertas por comuna (factor por clima)
    base_idx = {}
    for _, r in df_nat.iterrows():
        k = f"{pd.to_datetime(r['fecha_dt']).strftime('%Y-%m-%d')}T{int(r['hora']):02d}"
        base_idx[k] = float(r["pred_base"])

    items = []
    for serie in clima_series:
        comuna = serie["comuna"]
        n = min(len(serie["time"]), len(serie["precip"]), len(serie["prob"]), len(serie["wind"]))
        for i in range(n):
            iso = serie["time"][i]
            fecha, hour = to_local_date_hour(iso, TZ)
            key = f"{fecha}T{hour:02d}"

            base = int(round(base_idx.get(key, 0.0)))
            precip_mm = float(serie["precip"][i] or 0.0)
            prob_pct  = float(serie["prob"][i] or 0.0)
            wind_kmh  = float(serie["wind"][i] or 0.0)

            f = factor_clima(precip_mm, prob_pct, wind_kmh)
            ajust = int(max(0, round(base * f)))
            delta = ajust - base
            rel = (delta / base) if base > 0 else (1.0 if ajust > 0 else 0.0)

            if delta >= MIN_ABS_INCREASE and rel >= MIN_REL_INCREASE:
                items.append({
                    "fecha_dt": fecha,
                    "hora": hour,
                    "comuna": comuna,
                    "clima": etiqueta_clima(precip_mm, prob_pct, wind_kmh),
                    "factor": round(f, 3),
                    "llamadas_base": base,
                    "llamadas_ajustadas": ajust,
                    "llamadas_adicionales": int(delta)
                })

    items.sort(key=lambda x: (x["fecha_dt"], x["hora"], x["comuna"]))

    out = {
        "ok": True,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "horizon_days": HORIZON_DAYS,
        "thresholds": {
            "min_rel_increase": MIN_REL_INCREASE,
            "min_abs_increase": MIN_ABS_INCREASE
        },
        "items": items
    }

    os.makedirs("public", exist_ok=True)
    with open(OUT_ALERTAS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"✅ Alertas generadas: {OUT_ALERTAS_PATH} ({len(items)} filas)")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
