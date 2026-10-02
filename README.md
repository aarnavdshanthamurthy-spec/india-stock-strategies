# Indian stock swing strategies

**[stock_strategies.py](stock_strategies.py) contains all runtime Python code in one file**, including named functions for each stock, technical indicators, institutional screening, a deterministic backtester, position sizing, and a command-line runner.

The default research account is ₹10,000, with ₹100 planned risk per trade, a ₹4,000 maximum position, and up to two reserved positions. Strategies hold for at most 20–30 sessions. A 10% account-loss threshold prevents allocations when you supply the current account value. Stops can lose more than their intended amount after a gap.

These are **research hypotheses and paper-trading tools**, not guaranteed profitable strategies. Defining a strategy for a stock does not mean it passes the institutional, historical-data, or performance requirements. The code submits no broker orders.

## Run

Python 3.9 or newer:

```bash
python3 -m pip install -r requirements.txt
python3 stock_strategies.py --list
python3 stock_strategies.py --snapshot
python3 stock_strategies.py --run --symbols RADICO PREMIERENE ANTELOPUS IOLCP
python3 stock_strategies.py --run --discover
```

The default run includes all 11 explicitly requested stocks. `--discover` also measures 14 additional stock definitions against a price ceiling of ₹250, a 20-session mean volume of at least 1,000,000 shares, and mean turnover of at least ₹1 crore. This is a bounded discovery universe, not an exhaustive market scan. A low nominal price does not imply a cheap valuation or lower risk. Override `--max-price` and `--min-volume` to change the discovery screen.

The single Python file also embeds a dated research summary, printable with `--snapshot`. See [RESEARCH_SNAPSHOT.md](RESEARCH_SNAPSHOT.md) for the measured prices, volumes, institutional eligibility, and backtest outcomes from 2 October 2026. This snapshot is never used to authorize new trades.

## Individual Python functions

```python
import pandas as pd
from stock_strategies import (
    strategy_radico,
    strategy_premierene,
    strategy_antelopus,
    strategy_iolcp,
    strategy_hfcl,
    backtest_stock,
)

# Date,Open,High,Low,Close,Volume, using a consistent adjusted OHLC basis.
prices = pd.read_csv("output/prices/RADICO.NS.csv")
signals = strategy_radico(prices)
print(signals.tail())

# Optional alternate rule declared for this stock:
signals = strategy_radico(prices, variant="breakout20")
evaluation = backtest_stock("RADICO.NS", prices)
print(evaluation["reason"])
```

Each function returns a DataFrame with `enter_next_open`, `exit_next_open`, `atr14`, `stop_distance`, `target_distance`, and `max_holding_sessions`. A technical signal alone is not an order. No networking occurs when importing the file or calling a strategy with local prices.

| Stock | Python function | Default hypothesis | Alternate hypothesis |
|---|---|---|---|
| Anand Rathi Wealth | `strategy_anandrathi` | 10-session breakout | EMA50 recovery |
| Central Bank of India | `strategy_centralbk` | Trend pullback | 20-session breakout |
| Emcure Pharmaceuticals | `strategy_emcure` | EMA50 recovery | Trend pullback |
| GlaxoSmithKline Pharmaceuticals | `strategy_glaxo` | RSI35 rebound | Trend pullback |
| HFCL | `strategy_hfcl` | 20-session breakout | 10-session breakout |
| Ola Electric Mobility | `strategy_olaelec` | RSI35 rebound | 10-session breakout |
| Poonawalla Fincorp | `strategy_poonawalla` | EMA50 recovery | Trend pullback |
| Radico Khaitan | `strategy_radico` | Trend pullback | 20-session breakout |
| Premier Energies | `strategy_premierene` | 20-session breakout | EMA50 recovery |
| Antelopus Selan Energy | `strategy_antelopus` | Wider-stop breakout | EMA50 recovery |
| IOL Chemicals and Pharmaceuticals | `strategy_iolcp` | RSI35 rebound | 20-session breakout |

Additional named functions: `strategy_yesbank`, `strategy_suzlon`, `strategy_idea`, `strategy_pnb`, `strategy_bankindia`, `strategy_unionbank`, `strategy_ucobank`, `strategy_iob`, `strategy_nhpc`, `strategy_sjvn`, `strategy_irfc`, `strategy_idfcfirstb`, `strategy_gmrairport`, and `strategy_rpower`.

The requested “Antelopus solar energy” is interpreted as **Antelopus Selan Energy**, an oil and gas company. NSE confirms the [symbol change from SELAN to ANTELOPUS](https://nsearchives.nseindia.com/corporate/saman_16092025181426_SELAN_NameChange.pdf); see the [company description](https://antelopusenergy.com/about-us/). The code does not silently stitch pre-merger histories together. Short price history can prevent qualification.

## Exact strategy rules

All rules require positive volume, positive ATR, and the minimum rupee turnover. All except RSI rebound require close above EMA200 and EMA50 above EMA200. Volume comparisons use the **previous** 20 sessions.

| Rule | Entry at completed close | Stop distance | Profit target | Max sessions |
|---|---|---|---|---|
| `breakout20` | Above prior 20-session high, RSI14 50–80, volume ≥1.3× average | 2 ATR | 2.5R | 30 |
| `breakout10` | Above prior 10-session high, RSI14 50–80, volume ≥1.0× average | 1.5 ATR | 2.5R | 20 |
| `pullback` | Low touches EMA20, close above EMA20 and open, RSI14 40–65, volume ≥0.8× average | 2 ATR | 2R | 20 |
| `reclaim` | Close crosses above EMA50, RSI14 45–65, volume ≥1.0× average | 1.5 ATR | 2.5R | 30 |
| `rebound` | RSI14 crosses above 35 and close exceeds previous close, volume ≥0.8× average | 1.5 ATR | 2R | 20 |
| `wide_breakout` | Above prior 20-session high, RSI14 50–80, volume ≥1.5× average | 2.5 ATR | 2R | 30 |

R is the entry-to-stop distance. Entries occur at the next open with adverse slippage; a gap exceeding 0.5 ATR cancels the entry. Stop/target distances use signal-day ATR. All except `rebound` also exit at the next open after a close below EMA50. Countertrend rebound trades use price stops, targets and time exits. If one daily bar touches both stop and target, the stop wins. Gap-through stops fill at the opening price with slippage. Zero-volume bars cannot fill.

## Evidence and institutional ownership

The strict ownership rule requires both FII and DII percentages to rise by at least 0.10 percentage points after a flat/falling preceding quarter. Three consecutive quarters are required, and the newest quarter cannot be more than 130 days old. Percentage changes are an accumulation proxy, not proof of purchases on a particular day; issuance, buybacks and classification changes can affect them.

Public Screener tables do not reliably provide original publication timestamps. Default historical tests therefore evaluate **price rules**, and ownership is a **current screen**. Each requested stock still gets its strategy code and price-rule evaluation even if its ownership screen fails; only a passing ownership screen can produce a paper watch/signal status.

For dated institutional backtests, supply a CSV with `symbol,quarter_end,fii_pct,dii_pct,available_at,observed_at,source` and use `--institutional-backtest`. All compared rows need their original `available_at` dates. Do not backfill revised numbers with original publication dates. Current snapshots cannot establish historical availability.

```bash
python3 stock_strategies.py --run --symbols RADICO IOLCP \
  --price-dir data/prices --holdings-csv data/ownership.csv \
  --institutional-backtest --as-of 2026-10-01
```

CSV price files are named `SYMBOL.NS.csv`. All OHLC fields must share a consistent split/dividend-adjusted basis, with split-consistent volume. Missing, malformed, suspended or stale inputs are excluded. Provider throttling stops further ownership downloads and leaves their eligibility false.

The backtest uses 200 warmup bars, then 50% training, 25% validation and 25% final evaluation; at least 830 daily bars are required. Training needs 3+ trades and profit factor ≥1.0. Validation needs 3+ trades and profit factor ≥1.15. A rule is selected on validation return minus half its drawdown, then must pass a 5+ trade final period and doubled-slippage stress test. All stages must have positive net returns and accepted results must respect the 10% account drawdown limit. The runner does not replace a failed selected rule with a runner-up from the final period.

Historical periods and stocks have already been explored in this project, so these results are marked **exploratory, not independently validated**. A present-day universe, current ownership selection, and testing multiple rules create selection bias. Separate stock returns are not a combined portfolio return.

## Costs and outputs

Default research costs: 25 basis points of aggregate charges and 20 basis points slippage **per side**, plus ₹15.34 on sale as a reference DP charge. The fixed-charge example comes from [Zerodha’s published charges](https://zerodha.com/charges/); it is not a claim about your actual broker. Personal income/capital-gains taxes are excluded.

The run writes these local files under `output/`:

- `RESEARCH_REPORT.txt`: each stock’s rules, measured price/volume, ownership result and backtest results.
- `research.json`: complete metrics, equity curves and trade ledgers.
- `individual_strategies.py`: importable results; all executable strategy functions remain in `stock_strategies.py`.
- `prices/` and `ownership_snapshot.csv`: audit inputs. These caches are excluded from Git.

`order_specification(...)` can create an **unsubmitted paper-order specification** when ownership, backtests and price signals pass. It is not broker execution, and the caller must check the immediately following exchange session, quote freshness, cash and existing positions.

```bash
python3 stock_strategies.py --run --account-value 9000
python3 -m pytest tests -q
```

## Contributors

- [@ayaan-amodia](https://github.com/ayaan-amodia) — project contributor credit.

Credit does not grant collaborator permissions. This public repository is readable by everyone; no contributor write or admin access is granted by this README.
