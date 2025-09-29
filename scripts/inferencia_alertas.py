#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import datetime as dt

import numpy as np
import pandas as pd
import requests
from joblib import load

name: generar-alertas

on:
  workflow_dispatch:
  # 03:00 America/Santiago ≈ 06:00 UTC (ajusta si cambia el horario de verano)
  schedule:
    - cron: '0 6 * * *'

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      # (opcional, acelera pip)
      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      # (a veces LightGBM lo requiere; los runners suelen traerlo,
      # si te estorba lo puedes quitar)
      - name: Ensure libgomp
        run: sudo apt-get update && sudo apt-get install -y libgomp1

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run inference to generate alertas.json
        env:
          # URL del clima (7 días, por comuna) que consume tu script
          CLIMA_URL: "https://turnos-api-inbound.andres-eliecergonzalez.workers.dev/clima"
        run: |
          python scripts/inferencia_alertas.py

      - name: Commit & push changes
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "chore: actualizar alertas.json (automatizado)"
          commit_user_name: GitHub Action
          commit_user_email: actions@github.com
          file_pattern: public/alertas.json
