# Alerta de tráfico por clima (GitHub Actions)

Este repositorio genera `public/alertas.json` **cada día a las 03:00** (hora de Santiago) usando tu **modelo LightGBM** (.pkl) y el **clima a 7 días** por comuna desde tu URL.

## Estructura
```
public/
  (se genera) alertas.json
models/
  modelo_trafico_clima_lgbm_llamadas.pkl   <-- PON AQUÍ TU MODELO
  features.json                             <-- opcional
scripts/
  inferencia_alertas.py
data/
  semilla_llamadas.json                     <-- opcional (últimas 24–48h)
.github/workflows/
  alertas.yml
```

## Pasos para dejarlo andando
1. **Sube tu modelo** `modelo_trafico_clima_lgbm_llamadas.pkl` a `models/`.
   - (Opcional) sube `features.json` si quieres usar tu lista exacta de features.
2. En `.github/workflows/alertas.yml` **cambia** `CLIMA_FUTURO_URL` por tu endpoint real (p. ej. `https://mi-worker.workers.dev/clima`).
3. (Opcional) Si tienes datos recientes para iniciar lags, edita `data/semilla_llamadas.json`.
4. Haz **commit & push**. En **Actions** puedes lanzar el workflow manualmente (Run workflow) o esperar el cron (03:00).
5. Tu front puede consumir:
   ```
   https://raw.githubusercontent.com/<usuario>/<repo>/refs/heads/main/public/alertas.json
   ```

## Salida (`public/alertas.json`)
```json
{
  "ok": true,
  "generated_at": "2025-09-28T06:00:00Z",
  "horizon_days": 7,
  "thresholds": { "min_rel_increase": 0.1, "min_abs_increase": 10 },
  "items": [
    {
      "fecha_dt": "2025-09-29",
      "hora": 10,
      "comuna": "Puerto Montt",
      "clima": "lluvia_fuerte,alta_prob_precip",
      "factor": 1.26,
      "llamadas_base": 120,
      "llamadas_ajustadas": 151,
      "llamadas_adicionales": 31
    }
  ]
}
```

## Notas
- El modelo predice el **baseline nacional por hora** a 7 días sumando el **clima agregado nacional**. Luego aplica un **factor por comuna** (lluvia/viento/prob.) para detectar **picos locales**.
- Ajusta sensibilidad de alertas editando `MIN_REL_INCREASE` y `MIN_ABS_INCREASE` en el workflow o como variables de entorno.
- Si tu API de clima cambia de formato, adapta `parse_clima()` en `scripts/inferencia_alertas.py`.
