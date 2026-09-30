# VADER — Volatility of Abnormal Defence Equity Returns

**VADER** is a quantitative measure of geopolitical risk derived from the equity market. It is constructed as the exponentially weighted moving average (EWMA) conditional volatility of the market-filtered returns of a value-weighted portfolio of listed defence-exposed firms.

The measure was developed for the paper *Hutchinson, Thomas and Baur, Dirk G. and Trench, Allan and Hoang, Lai T., Always Tell Me the Odds: VADER, a Market-Implied Measure of Geopolitical Risk (September 30, 2026). Available at SSRN: https://ssrn.com/abstract=7543140 or http://dx.doi.org/10.2139/ssrn.7543140*. Please cite the paper when using this code or the VADER series. 

---

## Methodology (summary)

1. **Market filtering.** Each firm's daily returns are regressed on the S&P 500 over a 252-day rolling window ending at 
t−1. The day-t residual uses only information available at t−1.
2. **Portfolio aggregation.** Residuals are aggregated into a value-weighted portfolio. Membership updates annually from SIPRI; weights update monthly from market cap.
3. **Conditional volatility.** VADER is the EWMA standard deviation of the portfolio residual, with λ=0.94. Full detail, robustness, and interpretation are in the paper.

```text
VADER/
├── VADER/
│   ├── configuration/
│   │   └── config.py          # Ticker universe, SIPRI membership, LSEG config path
│   └── vader/
│       └── load_data.py       # VaderData class
├── example_usage.ipynb           # Example usage
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/pukkatee/VADER.git
cd VADER
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### LSEG configuration
VaderData retrieves all market data through lseg.data. Before use, set lseg_config_path in VADER/configuration/config.py to the path of your LSEG Workspace configuration, and ensure a valid LSEG Workspace session is available.
