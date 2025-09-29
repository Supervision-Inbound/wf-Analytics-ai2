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
# CONFIG (ajustable por env)
# =========================
PRED_URL   = os.getenv("PRED_URL",   "https://turnos-api-inbound.andres-eliecergonzalez.workers.dev/predicciones")
TURNOS_URL = os.getenv("TURNOS_URL", "https://turnos-api-inbound.andres-eliecergonzalez.workers.dev/turnos")

# salida
OUT_PATH = "public/alertas.json"

# ventana de exportación
DAYS_AHEAD = int(os.getenv("DAYS_AHEAD", "7"))
TZ_OFFSET_HOURS = int(os.getenv("TZ_OFFSET_HOURS", "3"))  # ~America/Santiago (aprox)

# Erlang-C / dimensionamiento
ASA_TARGET_SEC = float(os.getenv("ASA_TARGET_SEC", "22"))
SL_TARGET      = float(os.getenv("SL_TARGET", "0.90"))   # 90%
AHT_SEC        = float(os.getenv("AHT_SEC", "180"))      # fallback si no viene en predicción
PRODUCTIVIDAD  = float(os.getenv("PRODUCTIVIDAD", "0.85"))  # 85% => agentes efectivos = planificados * 0.85
INTERVAL_MIN   = int(os.getenv("INTERVAL_MIN", "60"))    # trabajamos por hora

# filtros de “alertas” (si quieres recortar ruido aquí)
CONFIDENCE_MIN = float(os.getenv("CONFIDENCE_MIN", "0.35"))
UMBRAL_FACTOR  = float(os.getenv("UMBRAL_FACTOR", "1.10"))
UMBRAL_MIN_EXTRA = int(os.getenv("UMBRAL_MIN_EXTRA", "5"))

# =========================
# Utilidades de tiempo
# =========================
def now_scl():
    # Runner en UTC; aproximamos SCL restando TZ_OFFSET_HOURS
    return dt.datetime.utcnow() - dt.timedelta(hours=TZ_OFFSET_HOURS)

def std_yyyy_mm_dd_h(dtobj: dt.datetime):
    return dtobj.strftime("%Y-%m-%d"), dtobj.hour

# =========================
# Fetch & parsing de predicciones (alertas inteligentes)
# =========================
def fetch_predicciones(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    # soporta lista o dict con 'rows'
    rows = data.get("rows", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise RuntimeError("Estructura de predicciones no reconocida")
    return rows

def filter_alertas_rows(rows):
    """Filtra a hoy -> +7 días y conserva sólo filas con impacto razonable."""
    t0 = now_scl().replace(minute=0, second=0, microsecond=0)
    tmax = t0 + dt.timedelta(days=DAYS_AHEAD)

    out = []
    for r in rows:
        # tolera distintos nombres de campos
        fecha = r.get("fecha") or r.get("fecha_dt") or r.get("date")
        hora  = r.get("hora")  if "hora" in r else r.get("hour")
        comuna = r.get("comuna") or r.get("name")

        # parse fecha/hora
        try:
            h = int(hora)
        except:
            continue
        try:
            d = pd.to_datetime(fecha, errors="coerce")
        except:
            d = None
        if d is None or pd.isna(d):
            continue
        when = dt.datetime(d.year, d.month, d.day, h)

        if not (t0 <= when <= tmax):
            continue

        # campos core (con tolerancia a nombres)
        expected = r.get("expected_calls") or r.get("pred_llamadas_base") or r.get("llamadas_base") or r.get("pred_llamadas")
        total    = r.get("total_calls")    or r.get("pred_llamadas_ajustada") or r.get("llamadas_total") or r.get("pred")
        extra    = r.get("extra_calls")
        factor   = r.get("factor_ajustado") or r.get("factor") or (float(total)/float(expected) if expected and total else 1.0)
        conf     = r.get("confidence") or r.get("confianza") or 0.5
        evento   = r.get("evento") or r.get("tipo") or "ninguno"
        wind     = r.get("wind_speed_10m") or r.get("viento") or None
        rain_mm  = r.get("precip_mm") or r.get("lluvia_mm") or None
        rain_p   = r.get("precip_prob") or r.get("prob_lluvia") or None

        # si falta total/expected, intenta construirlos
        if expected is None and total is not None:
            expected = total / max(1e-9, factor)
        if total is None and expected is not None:
            total = expected * max(1.0, float(factor))

        # filtrar ruido de alertas
        if (float(conf) < CONFIDENCE_MIN) or (float(factor) < UMBRAL_FACTOR):
            continue
        if extra is None and expected is not None and total is not None:
            cand_extra = int(round(float(total) - float(expected)))
            if cand_extra < max(UMBRAL_MIN_EXTRA, int(math.ceil(0.08*float(expected)))):
                continue
        elif extra is not None and int(extra) < UMBRAL_MIN_EXTRA:
            continue

        out.append({
            "fecha": when.strftime("%Y-%m-%d"),
            "hora":  when.hour,
            "comuna": comuna,
            "evento": evento,
            "wind_speed_10m": round(float(wind), 2) if wind is not None else None,
            "precip_mm": round(float(rain_mm), 2) if rain_mm is not None else None,
            "precip_prob": int(round(float(rain_p))) if rain_p is not None else None,
            "expected_calls": int(round(float(expected))) if expected is not None else None,
            "extra_calls":    int(round(float(extra)))    if extra is not None else int(round(float(total) - float(expected))) if (expected is not None and total is not None) else None,
            "total_calls":    int(round(float(total)))    if total is not None else None,
            "factor_ajustado": round(float(factor), 3) if factor is not None else None,
            "confidence": round(float(conf), 3)
        })
    return out

# =========================
# Erlang-C helpers
# =========================
def erlang_c_probability(A, N):
    """
    Probabilidad de espera (cola) con carga A (erlangs) y N agentes.
    Fórmula clásica de Erlang C.
    """
    if N <= 0 or A <= 0:
        return 1.0
    if A >= N:  # sistema inestable => prob cola  ~1
        return 1.0
    # sumar términos 0..N-1
    sum_terms = 0.0
    term = 1.0
    for k in range(1, N):
        term *= A / k
        sum_terms += term
    # término de estado saturado
    termN = term * (A / N)
    p0 = 1.0 / (1.0 + sum_terms + termN * (1.0 / (1.0 - A / N)))
    pw = termN * p0 * (1.0 / (1.0 - A / N))
    return pw

def service_level(A, N, asa_sec):
    """
    NS = 1 - P(espera > asa) = 1 - ErlangC*exp(-(N-A)*asa/AHT)
    con tasa servicio = 1/AHT (A = lambda * AHT)
    """
    if N <= 0 or A <= 0:
        return 0.0
    if A >= N:
        return 0.0
    pw = erlang_c_probability(A, N)
    mu = 1.0  # trabajaremos con A ya como erlangs (= lambda*AHT). Por eso mu=1 y asa multiplica (N-A)
    # En la forma normalizada, la “tasa de salida” efectiva para espera es (N - A) * mu / A
    # como A = λ * AHT, y mu = 1/AHT, se simplifica a exp(-(N-A)*asa/A)
    return 1.0 - pw * math.exp(-(N - A) * (asa_sec / AHT_SEC) * (AHT_SEC / A))  # se simplifica a exp(-(N-A)*asa/A)

def required_agents_by_erlang(lambda_calls, aht_sec, asa_target, sl_target):
    """
    Dado λ (llamadas/seg) y AHT (seg), carga A=λ*AHT. Subimos N hasta cumplir NS>=sl_target.
    """
    if lambda_calls <= 0:
        return 0
    A = lambda_calls * aht_sec
    N = max(1, int(math.ceil(A)))  # mínimo estable
    # búsqueda incremental
    for n in range(N, N + 1000):
        ns = service_level(A, n, asa_target)
        if ns >= sl_target:
            return n
    return N + 1000  # fallback teórico

# =========================
# Turnos → agentes planificados por hora
# =========================
def excel_serial_to_date(serial):
    # Excel date (base 1899-12-30); la API ya normaliza, pero por si acaso
    if not isinstance(serial, (int, float)) or serial <= 0:
        return None
    return dt.datetime(1899, 12, 30) + dt.timedelta(days=float(serial))

def excel_fraction_to_time(frac):
    # Fracción de día → (h, m)
    if not isinstance(frac, (int, float)) or frac < 0:
        return None
    total_min = round(float(frac) * 24 * 60)
    h = (total_min // 60) % 24
    m = total_min % 60
    return h, m

def parse_turnos(raw):
    """
    Acepta lista o dict con 'value'. Espera columnas tipo Excel:
      Fecha (serial), Ini (fracción), Fin (fracción) o equivalentes.
    Devuelve lista de turnos con inicio/fin datetime.
    """
    data = raw if isinstance(raw, list) else raw.get("value", [])
    out = []
    for row in data:
        fecha = row.get("Fecha") or row.get("fecha") or row.get("Date")
        ini   = row.get("Ini")   or row.get("ini")   or row.get("INICIO")
        fin   = row.get("FIN")   or row.get("Fin")   or row.get("fin")
        d = excel_serial_to_date(fecha) if isinstance(fecha, (int, float)) else None
        if d is None:
            continue
        s_hm = excel_fraction_to_time(ini) if ini is not None else None
        e_hm = excel_fraction_to_time(fin) if fin is not None else None
        if not s_hm or not e_hm:
            continue
        start = d.replace(hour=s_hm[0], minute=s_hm[1], second=0, microsecond=0)
        end   = d.replace(hour=e_hm[0], minute=e_hm[1], second=0, microsecond=0)
        if end <= start:
            end += dt.timedelta(days=1)  # turno cruzando medianoche
        out.append({"ini": start, "fin": end})
    return out

def hourly_staff_from_shifts(turnos, t0, tmax):
    """
    Cuenta agentes planificados por cada hora entre [t0, tmax].
    Un turno contribuye a cada hora en la que está activo (intersección > 0 min).
    """
    buckets = defaultdict(int)
    cur = t0
    while cur <= tmax:
        buckets[cur] = 0
        cur += dt.timedelta(hours=1)
    for sh in turnos:
        ini, fin = sh["ini"], sh["fin"]
        # recorta a ventana
        s = max(ini, t0)
        e = min(fin, tmax + dt.timedelta(hours=1))
        if e <= s:
            continue
        # marcar horas tocadas
        h = dt.datetime(s.year, s.month, s.day, s.hour)
        while h < e:
            # si hay solapamiento con la hora [h, h+1)
            ovl = max(0, (min(e, h + dt.timedelta(hours=1)) - max(s, h)).total_seconds())
            if ovl > 0:
                buckets[h] += 1
            h += dt.timedelta(hours=1)
    # a lista ordenada
    out = []
    for when in sorted(buckets.keys()):
        out.append({"when": when, "agents_planned": buckets[when]})
    return out

# =========================
# Main
# =========================
def main():
    # 1) Predicciones (alertas inteligentes)
    pred_rows = fetch_predicciones(PRED_URL)
    alertas = filter_alertas_rows(pred_rows)

    # 2) Forecast por hora (sumamos total_calls por fecha-hora en todas las comunas)
    t0 = now_scl().replace(minute=0, second=0, microsecond=0)
    tmax = t0 + dt.timedelta(days=DAYS_AHEAD)

    # intentamos usar AHT si viene en las filas; si no, usamos AHT_SEC global
    df_alert = pd.DataFrame(alertas)
    if df_alert.empty:
        df_alert = pd.DataFrame(columns=["fecha", "hora", "total_calls"])

    df_alert["dt"] = pd.to_datetime(df_alert["fecha"]) + pd.to_timedelta(df_alert["hora"], unit="h")
    df_alert = df_alert[(df_alert["dt"] >= t0) & (df_alert["dt"] <= tmax)].copy()

    # total de llamadas por hora (todas las comunas)
    calls_by_hour = (
        df_alert.groupby("dt", as_index=False)["total_calls"]
        .sum()
        .rename(columns={"total_calls": "calls"})
    )
    # AHT por hora (si alguna fila lo trae; si no, global)
    if "pred_tmo_min" in df_alert.columns:
        # pred_tmo_min viene en minutos → a segundos (promedio por hora)
        aht_by_hour = (
            df_alert.groupby("dt", as_index=False)["pred_tmo_min"]
            .mean()
            .assign(aht_sec=lambda d: d["pred_tmo_min"] * 60.0)[["dt", "aht_sec"]]
        )
    else:
        aht_by_hour = pd.DataFrame({"dt": calls_by_hour["dt"], "aht_sec": AHT_SEC})

    demand = calls_by_hour.merge(aht_by_hour, on="dt", how="left")
    demand["aht_sec"] = demand["aht_sec"].fillna(AHT_SEC)

    # 3) Requeridos por Erlang C
    req_records = []
    for _, row in demand.iterrows():
        when = row["dt"].to_pydatetime()
        calls = float(row["calls"] or 0.0)
        aht   = float(row["aht_sec"] or AHT_SEC)

        # tasa λ (llamadas/seg) en el intervalo de 1 hora
        lam = calls / (INTERVAL_MIN * 60.0)
        n_req = required_agents_by_erlang(lam, aht, ASA_TARGET_SEC, SL_TARGET)
        req_records.append({"when": when, "calls": int(round(calls)), "aht_sec": aht, "agents_required": int(n_req)})

    df_req = pd.DataFrame(req_records) if req_records else pd.DataFrame(columns=["when","calls","aht_sec","agents_required"])

    # 4) Turnos → agentes planificados
    try:
        tr = requests.get(TURNOS_URL, timeout=60); tr.raise_for_status()
        turnos_raw = tr.json()
        shifts = parse_turnos(turnos_raw)
    except Exception as e:
        shifts = []

    staff = hourly_staff_from_shifts(shifts, t0, tmax) if shifts else [{"when": w, "agents_planned": 0} for w in pd.date_range(t0, tmax, freq="1H")]

    df_staff = pd.DataFrame(staff)
    df = df_req.merge(df_staff, on="when", how="left").fillna({"agents_planned": 0})

    # productividad: dotación efectiva
    df["agents_planned_eff"] = (df["agents_planned"] * PRODUCTIVIDAD).round(2)
    df["deficit"] = (df["agents_required"] - df["agents_planned_eff"]).round(2)

    # 5) Salida “staffing_deficit”
    staffing_deficit = []
    for _, r in df.sort_values("when").iterrows():
        fecha, hora = std_yyyy_mm_dd_h(r["when"])
        staffing_deficit.append({
            "fecha": fecha,
            "hora": int(hora),
            "calls": int(r["calls"]),
            "aht_sec": round(float(r["aht_sec"]), 2),
            "asa_target_sec": ASA_TARGET_SEC,
            "sl_target": SL_TARGET,
            "agents_required": int(r["agents_required"]),
            "agents_planned": int(r["agents_planned"]),
            "agents_planned_eff": float(r["agents_planned_eff"]),
            "deficit": float(r["deficit"])  # >0 => faltan agentes
        })

    # 6) Escribir JSON consolidado
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "alertas": alertas,
            "staffing_deficit": staffing_deficit
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ Generado {OUT_PATH} | alertas={len(alertas)} | filas staffing={len(staffing_deficit)}")

if __name__ == "__main__":
    main()

