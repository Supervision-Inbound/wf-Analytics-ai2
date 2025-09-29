#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera public/alertas.json usando el clima de los próximos 7 días
y una línea base de llamadas por hora. Se enfoca en alertas a partir
de HOY hacia adelante (no mete fechas pasadas).

- Toma CLIMA_URL desde la variable de entorno (o usa el valor por defecto)
- Umbrales configurables abajo
- Si data/semilla_llamadas.json existe, usa su perfil horario como baseline
- Si no hay semilla, usa un baseline plano (constante)

Salida: public/alertas.json
"""

import os
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests


# ========= CONFIG =========
DEFAULT_CLIMA_URL = "https://turnos-api-inbound.andres-eliecergonzalez.workers.dev/clima"

# Umbrales (ajústalos a tu realidad)
TH_PPT_PROB = 60      # % prob. precipitación
TH_PPT_MM   = 0.5     # mm/h
TH_WIND     = 35      # km/h

# Factores de alza por tipo (aprox, puedes calibrarlos)
FACTOR_LLUVIA = 0.15   # +15% sobre baseline
FACTOR_VIENTO = 0.10   # +10% sobre baseline

# Mapeo simple de normalización de nombres (puedes extenderlo)
COMUNA_FIX = {
    "pto. montt": "Puerto Montt",
    "pto montt": "Puerto Montt",
    "san juan de la costa": "San Juan de la Costa",
    "valdivia": "Valdivia",
    "osorno": "Osorno",
    # agrega más si lo necesitas
}


# ========= Utils =========
def normalize_comuna(name: str) -> str:
    if not isinstance(name, str):
        return ""
    key = name.strip().lower()
    return COMUNA_FIX.get(key, name.strip())


def load_seed_baseline():
    """
    Lee data/semilla_llamadas.json si existe. Acepta:
      - dict con clave "perfil_horario": { "0": n0, ..., "23": n23 }
      - dict con clave "perfil_diario_hora": { "0": {...}, "1": {...}, ... } (L-M-...-D)
      - lista de 24 valores (perfil horario simple)
    Si nada existe, devuelve un baseline plano = 100.
    """
    path = "data/semilla_llamadas.json"
    if not os.path.exists(path):
        # baseline plano
        perfil_horario = {str(h): 100.0 for h in range(24)}
        return perfil_horario, None

    with open(path, "r", encoding="utf-8") as f:
        seed = json.load(f)

    # Caso lista [24]
    if isinstance(seed, list) and len(seed) == 24:
        return {str(i): float(v) for i, v in enumerate(seed)}, None

    # Caso dict con perfil_horario
    if isinstance(seed, dict) and "perfil_horario" in seed:
        ph = seed["perfil_horario"]
        # asegurar str keys
        perfil_horario = {str(k): float(v) for k, v in ph.items()}
        perfil_diario = seed.get("perfil_diario_hora", None)
        return perfil_horario, perfil_diario

    # Caso dict con perfil_diario_hora directamente
    if isinstance(seed, dict) and "perfil_diario_hora" in seed:
        perfil_diario = seed["perfil_diario_hora"]
        # construir un perfil horario promedio si no viene
        medias = {}
        for h in range(24):
            vals = []
            for d in perfil_diario.values():
                if str(h) in d:
                    try:
                        vals.append(float(d[str(h)]))
                    except Exception:
                        pass
            medias[str(h)] = float(np.mean(vals)) if vals else 100.0
        return medias, perfil_diario

    # Fallback
    perfil_horario = {str(h): 100.0 for h in range(24)}
    return perfil_horario, None


def baseline_calls_for(dt_local: datetime, perfil_horario: dict, perfil_diario: dict | None) -> float:
    """
    Devuelve baseline esperado para una fecha/hora específica.
    Usa perfil_diario_hora si está disponible; si no, usa perfil_horario.
    """
    h = str(dt_local.hour)
    if perfil_diario is not None:
        # Monday=0 ... Sunday=6
        dow = str(dt_local.weekday())
        if dow in perfil_diario and h in perfil_diario[dow]:
            try:
                return float(perfil_diario[dow][h])
            except Exception:
                pass
    # fallback
    return float(perfil_horario.get(h, 100.0))


def fetch_clima(url: str) -> list[dict]:
    """
    Espera estructura tipo:
    [
      {
        "name": "Puerto Montt",
        "lat": -41.45,
        "lon": -72.94,
        "timezone": "...",
        "hourly": {
          "time": [... ISO strings ...],
          "wind_speed_10m": [...],
          "precipitation": [...],
          "precipitation_probability": [...]
        }
      },
      ...
    ]
    """
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "rows" in data:
        # por si viene envuelto
        return data["rows"]
    if isinstance(data, list):
        return data
    return []


def build_alerts(clima_rows: list[dict], perfil_horario: dict, perfil_diario: dict | None) -> list[dict]:
    """
    Genera lista de alertas SOLO desde HOY (inicio del día local) hacia +7 días.
    Cada alerta es por (fecha, hora, comuna, evento) con baseline y extra_calls.
    """
    now_utc = datetime.now(timezone.utc)
    # Definimos el inicio de "hoy" en hora local de Chile (America/Santiago ~ UTC-3/UTC-4).
    # Para evitar dependencias de zoneinfo, aproximamos con offset actual del feed (viene en ISO local).
    # El feed trae timestamps ISO local? Si viene en ISO con offset, abajo parseamos como naive local.
    today_local = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff_ini = today_local
    cutoff_end = today_local + timedelta(days=7)

    out = []

    for row in clima_rows:
        cname = normalize_comuna(row.get("name"))
        hourly = (row.get("hourly") or {})
        times = hourly.get("time") or []
        wind = hourly.get("wind_speed_10m") or []
        ppt = hourly.get("precipitation") or []
        pprob = hourly.get("precipitation_probability") or []

        n = min(len(times), len(wind), len(ppt), len(pprob))
        if n == 0:
            continue

        for i in range(n):
            # Parseo hora local (string tipo "2025-10-01T03:00")
            try:
                t_local = datetime.fromisoformat(str(times[i]))
            except Exception:
                # si no parsea, saltar
                continue

            if t_local < cutoff_ini or t_local > cutoff_end:
                continue

            w = float(wind[i] if wind[i] is not None else 0.0)
            pr = float(pprob[i] if pprob[i] is not None else 0.0)
            mm = float(ppt[i] if ppt[i] is not None else 0.0)

            eventos = []
            if pr >= TH_PPT_PROB and mm >= TH_PPT_MM:
                eventos.append(("lluvia", FACTOR_LLUVIA))
            if w >= TH_WIND:
                eventos.append(("viento", FACTOR_VIENTO))

            if not eventos:
                continue  # no dispara alerta

            # baseline esperado para esa hora
            base = baseline_calls_for(t_local, perfil_horario, perfil_diario)

            # combinar factores (si hay lluvia y viento a la vez, sumamos factores)
            factor_total = 0.0
            tipos = []
            for ev, fac in eventos:
                factor_total += fac
                tipos.append(ev)

            extra = max(0, round(base * factor_total))
            total = int(max(0, round(base + extra)))

            out.append({
                "fecha": t_local.strftime("%Y-%m-%d"),
                "hora": t_local.hour,
                "comuna": cname,
                "evento": "+".join(tipos),  # "lluvia", "viento" o "lluvia+viento"
                "wind_speed_10m": w,
                "precip_mm": mm,
                "precip_prob": pr,
                "expected_calls": int(round(base)),
                "extra_calls": int(extra),
                "total_calls": total
            })

    # Ordenar por fecha, hora, comuna
    out.sort(key=lambda r: (r["fecha"], r["hora"], r["comuna"]))
    return out


def ensure_dirs():
    os.makedirs("public", exist_ok=True)


def main():
    ensure_dirs()

    clima_url = os.environ.get("CLIMA_URL", DEFAULT_CLIMA_URL)

    # 1) baseline
    perfil_horario, perfil_diario = load_seed_baseline()

    # 2) clima futuro (7 días)
    rows = fetch_clima(clima_url)

    # 3) alertas (solo desde HOY en adelante)
    alertas = build_alerts(rows, perfil_horario, perfil_diario)

    # 4) dump
    out_path = "public/alertas.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(alertas, f, indent=2, ensure_ascii=False)

    print(f"✅ Generado {out_path} con {len(alertas)} alertas")


if __name__ == "__main__":
    main()
