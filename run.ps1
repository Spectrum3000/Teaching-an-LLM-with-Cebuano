$ErrorActionPreference = 'Stop'
if (-not (Test-Path '.venv')) { py -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\step0_baseline.py
.\.venv\Scripts\python.exe src\prepare_corpus.py
.\.venv\Scripts\python.exe src\step1_lora.py --epochs 1 --max-steps 5
Write-Host 'Smoke run complete. Increase --max-steps for the real training run.'
