import json
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / 'all_tickers.json') as f:
    all_tickers=json.load(f)

with open(BASE_DIR / 'lseg_rics_by_year.json') as f:
    rics_by_year=json.load(f)

with open(BASE_DIR / 'config.yml', 'r') as file:
    config = yaml.safe_load(file)


