#!/usr/bin/env python3
"""Indian stock swing strategies — all runtime code in one file.

Research only: rules and measured backtests do not guarantee future profits.
Each strategy_SYMBOL function returns technical signals, not a broker order.
Defaults: Rs 10,000 capital, Rs 100 planned risk/trade, 40% position cap.

Examples:
    python stock_strategies.py --list
    python stock_strategies.py --run --symbols RADICO PREMIERENE ANTELOPUS IOLCP
    python stock_strategies.py --run --discover --as-of 2026-10-02
    from stock_strategies import strategy_radico, strategy_hfcl, backtest_stock

Dependencies: numpy, pandas, requests, beautifulsoup4, yfinance.
"""
from __future__ import annotations


# ===========================================================================
# DATA INPUTS
# ===========================================================================

import io
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def symbol_name(value):
    if pd.isna(value):
        raise ValueError('Stock symbol cannot be missing')
    value = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z0-9&_.-]+", value) or '..' in value:
        raise ValueError(f"Invalid stock symbol: {value!r}")
    return value if value.endswith(('.NS', '.BO')) else value + '.NS'


def get_text(url):
    response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()
    return response.text


def nifty500():
    url = 'https://archives.nseindia.com/content/indices/ind_nifty500list.csv'
    return [symbol_name(x) for x in pd.read_csv(io.StringIO(get_text(url)))['Symbol']]


def parse_shareholding(html, symbol, observed_at):
    """Parse only quarterly tables. Missing FII/DII is unknown, never zero."""
    soup = BeautifulSoup(html, 'html.parser')
    section = soup.select_one('#shareholding')
    if section is None:
        raise ValueError('No public shareholding table')
    tables = section.select('table')
    records = []
    for table in tables:
        headings = [x.get_text(' ', strip=True) for x in table.select('thead th')]
        if not headings:
            continue
        dates = []
        for heading in headings[1:]:
            try:
                date = pd.to_datetime(heading, format='%b %Y') + pd.offsets.MonthEnd(0)
            except ValueError:
                date = pd.NaT
            dates.append(date)
        # Yearly tables and irregular capital-change snapshots are not quarter comparisons.
        valid_dates = [d for d in dates if pd.notna(d) and d.month in (3, 6, 9, 12)]
        if len(valid_dates) < 3 or not any((b - a).days < 110 for a, b in zip(valid_dates, valid_dates[1:])):
            continue
        values = {}
        for row in table.select('tbody tr'):
            cells = row.select('td')
            if not cells:
                continue
            label = cells[0].get_text(' ', strip=True).replace('+', '').strip()
            if label in ('FIIs', 'DIIs'):
                values[label] = [pd.to_numeric(x.get_text(strip=True).replace('%', '').replace(',', ''), errors='coerce') for x in cells[1:]]
        if set(values) != {'FIIs', 'DIIs'}:
            continue
        for i, date in enumerate(dates):
            if pd.isna(date) or date.month not in (3, 6, 9, 12):
                continue
            if date > pd.Timestamp(observed_at):
                # Month labels can include an in-progress/irregular disclosure.
                continue
            if any(i >= len(values[k]) for k in values):
                continue
            records.append(dict(symbol=symbol, quarter_end=date, fii_pct=values['FIIs'][i],
                                dii_pct=values['DIIs'][i], available_at=pd.NaT,
                                observed_at=observed_at,
                                source=f'https://www.screener.in/company/{quote(symbol[:-3])}/consolidated/'))
        break
    if not records:
        raise ValueError('No comparable quarterly FII/DII observations')
    return validate_holdings(pd.DataFrame(records))


def public_holdings(symbol, observed_at):
    if not symbol.endswith('.NS'):
        raise ValueError('Public ownership downloader requires NSE symbols; use CSV for BSE')
    url = f'https://www.screener.in/company/{quote(symbol[:-3])}/consolidated/'
    return parse_shareholding(get_text(url), symbol, observed_at)


def validate_holdings(frame):
    required = {'symbol', 'quarter_end', 'fii_pct', 'dii_pct', 'source'}
    if not required.issubset(frame):
        raise ValueError(f'Ownership CSV missing columns: {sorted(required - set(frame))}')
    frame = frame.copy()
    frame['symbol'] = frame.symbol.map(symbol_name)
    for col in ('quarter_end', 'available_at', 'observed_at'):
        if col not in frame:
            frame[col] = pd.NaT
        frame[col] = pd.to_datetime(frame[col], errors='raise').dt.normalize()
    if frame.quarter_end.isna().any():
        raise ValueError('quarter_end cannot be missing')
    if not frame.quarter_end.dt.is_quarter_end.all():
        raise ValueError('Use calendar quarter-end observations only')
    if frame[['available_at', 'observed_at']].isna().all(axis=1).any():
        raise ValueError('Each ownership row needs available_at or observed_at')
    if (frame.available_at < frame.quarter_end).any() or (frame.observed_at < frame.quarter_end).any():
        raise ValueError('Ownership cannot be known before quarter end')
    for col in ('fii_pct', 'dii_pct'):
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if not np.isfinite(frame[col]).all() or not frame[col].between(0, 100).all():
            raise ValueError(f'{col} must contain percentages from 0 to 100, without blanks')
    if ((frame.fii_pct + frame.dii_pct) > 100.01).any():
        raise ValueError('FII + DII ownership exceeds 100%')
    if frame.source.isna().any() or frame.source.astype(str).str.strip().eq('').any():
        raise ValueError('Ownership source is required')
    if frame.duplicated(['symbol', 'quarter_end']).any():
        raise ValueError('Duplicate/revised ownership quarters: supply a point-in-time, unrevised series')
    return frame.sort_values(['symbol', 'quarter_end']).reset_index(drop=True)


def ownership_signal(frame, as_of, mode='both_new', min_increase=0.10, max_age=130, historical=False):
    """New = material rise after flat/falling previous quarter; units are percentage points.

    Ownership changes are accumulation proxies, not evidence of transaction dates.
    Historical mode requires original publication dates for every compared record.
    """
    known = frame.available_at if historical else frame.available_at.fillna(frame.observed_at)
    rows = frame.loc[(known <= as_of) & (frame.quarter_end <= as_of)].sort_values('quarter_end').tail(3)
    if len(rows) < 3:
        return {'eligible': False, 'reason': 'Need three known quarterly observations'}
    dates = rows.quarter_end
    if not dates.diff().dropna().dt.days.between(80, 100).all():
        return {'eligible': False, 'reason': 'Nonconsecutive quarterly observations'}
    age = (as_of - dates.iloc[-1]).days
    if age > max_age:
        return {'eligible': False, 'reason': 'Latest ownership quarter is stale'}
    details = {}
    for kind in ('fii', 'dii'):
        values = rows[f'{kind}_pct'].to_numpy()
        delta, previous = float(values[2] - values[1]), float(values[1] - values[0])
        details[f'{kind}_pct'] = float(values[2])
        details[f'{kind}_delta_pp'] = round(delta, 6)
        details[f'{kind}_previous_delta_pp'] = round(previous, 6)
        details[f'{kind}_new_buying'] = bool(delta >= min_increase - 1e-9 and previous <= 1e-9)
    rising = details['fii_delta_pp'] >= min_increase and details['dii_delta_pp'] >= min_increase
    if mode == 'both_new':
        eligible = details['fii_new_buying'] and details['dii_new_buying']
    elif mode == 'either_new_both_rising':
        eligible = rising and (details['fii_new_buying'] or details['dii_new_buying'])
    else:
        raise ValueError(f'Unknown ownership mode: {mode}')
    return dict(eligible=bool(eligible), reason='New accumulation proxy' if eligible else 'No new accumulation under requested rule',
                quarter_end=str(dates.iloc[-1].date()), age_days=age,
                available_at=None if pd.isna(rows.available_at.iloc[-1]) else str(rows.available_at.iloc[-1].date()),
                source=str(rows.source.iloc[-1]), **details)


def validate_prices(frame):
    frame = frame.copy()
    if 'Date' in frame:
        frame = frame.set_index('Date')
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    if frame.index.has_duplicates or frame.index.isna().any():
        raise ValueError('Duplicate or missing price dates')
    frame = frame.sort_index()
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not set(cols).issubset(frame):
        raise ValueError(f'Prices need {cols}')
    frame = frame[cols].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Nonfinite/missing OHLCV data')
    if (frame[cols[:4]] <= 0).any().any() or (frame.Volume < 0).any():
        raise ValueError('Nonpositive prices or negative volume')
    if ((frame.High < frame[['Open', 'Close', 'Low']].max(axis=1)) |
            (frame.Low > frame[['Open', 'Close', 'High']].min(axis=1))).any():
        raise ValueError('Invalid OHLC ranges')
    if frame.empty:
        raise ValueError('No price history')
    return frame


def load_prices(symbol, start, as_of, price_dir=None):
    if price_dir:
        return validate_prices(pd.read_csv(Path(price_dir) / f'{symbol}.csv')).loc[start:as_of]
    import yfinance as yf
    frame = yf.Ticker(symbol).history(start=str(start.date()), end=str((as_of + pd.Timedelta(days=1)).date()),
                                      auto_adjust=True, actions=False, raise_errors=True)
    return validate_prices(frame).loc[start:as_of]

# ===========================================================================
# SIGNALS AND BACKTEST ENGINE
# ===========================================================================

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Strategy:
    name: str
    family: str
    lookback: int
    stop_atr: float
    reward_risk: float
    max_hold: int
    volume_multiple: float
    trend_exit_ema: int = 50


# Fixed, deliberately small hypothesis set. The untouched test cannot select a runner-up.
STRATEGIES = (
    Strategy('breakout_20', 'breakout', 20, 2.0, 2.5, 30, 1.3),
    Strategy('breakout_55', 'breakout', 55, 2.5, 3.0, 50, 1.2),
    Strategy('trend_pullback', 'pullback', 20, 2.0, 2.0, 20, 0.8),
)


@dataclass(frozen=True)
class Risk:
    capital: float = 100000.0
    risk_fraction: float = 0.005
    allocation: float = 0.20
    fee_bps: float = 25.0
    slippage_bps: float = 20.0
    min_turnover: float = 10000000.0
    max_participation: float = 0.005
    max_gap_atr: float = 0.5
    fixed_exit_fee: float = 0.0


def features(prices):
    f = prices.copy()
    close = f.Close
    for span in (20, 50, 200):
        f[f'ema{span}'] = close.ewm(span=span, adjust=False, min_periods=span).mean()
    true_range = pd.concat([f.High - f.Low, (f.High - close.shift()).abs(),
                            (f.Low - close.shift()).abs()], axis=1).max(axis=1)
    f['atr'] = true_range.rolling(14).mean()
    changes = close.diff()
    gain = changes.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = -changes.clip(upper=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    f['rsi'] = (100 - 100 / (1 + gain / loss)).where(loss != 0, 100).where((gain + loss) != 0, 50)
    f['volume_reference'] = f.Volume.shift().rolling(20).mean()
    f['turnover'] = (close * f.Volume).rolling(20).mean()
    for window in (10, 20, 55):
        f[f'high{window}'] = f.High.shift().rolling(window).max()
    return f


def entry_signal(f, strategy, risk):
    trend = (f.Close > f.ema200) & (f.ema50 > f.ema200)
    liquid = (f.turnover >= risk.min_turnover) & (f.Volume > 0)
    common = liquid & (f.atr > 0) & (f.Volume >= f.volume_reference * strategy.volume_multiple)
    if strategy.family == 'breakout':
        setup = trend & (f.Close > f[f'high{strategy.lookback}']) & f.rsi.between(50, 80)
    elif strategy.family == 'pullback':
        setup = ((f.Low <= f.ema20) & (f.Close > f.ema20) &
                 (f.Close > f.Open) & f.rsi.between(40, 65) & trend)
    elif strategy.family == 'ema_reclaim':
        setup = (trend & (f.Close > f.ema50) & (f.Close.shift() <= f.ema50.shift())
                 & f.rsi.between(45, 65))
    elif strategy.family == 'rsi_rebound':
        # Deliberately permits countertrend trades; time/price stops govern the exit.
        setup = (f.rsi > 35) & (f.rsi.shift() <= 35) & (f.Close > f.Close.shift())
    else:
        raise ValueError(f'Unknown strategy family: {strategy.family}')
    return (common & setup).fillna(False)


def position_size(equity, entry, distance, volume, risk):
    fee = risk.fee_bps / 10000
    slip = risk.slippage_bps / 10000
    # Includes estimated round-trip charges and adverse stop slippage in risk sizing.
    loss_per_share = distance + entry * (2 * fee + slip)
    if min(equity, entry, distance, volume, loss_per_share) <= 0:
        return 0
    return max(0, math.floor(min((equity * risk.risk_fraction - risk.fixed_exit_fee) / loss_per_share,
                                equity * risk.allocation / (entry * (1 + fee)),
                                volume * risk.max_participation)))


def simulate(f, strategy, risk, start, end, institutional_mask=None):
    """Signals at close t fill at open t+1. end is exclusive; every segment starts flat.

    Stop wins on ambiguous bars, gap stops fill at open, targets never get a favorable
    gap fill. Zero-volume bars cannot execute. Terminal liquidation requires volume.
    """
    if not 1 <= start < end <= len(f):
        raise ValueError('Backtest bounds require 1 <= start < end <= number of bars')
    signals = entry_signal(f, strategy, risk)
    if institutional_mask is not None:
        signals &= institutional_mask.reindex(f.index, fill_value=False)
    cash = risk.capital
    position = None
    trades, equity, dates = [], [cash], []
    fee, slip = risk.fee_bps / 10000, risk.slippage_bps / 10000

    def close_position(reference, date, reason):
        nonlocal cash, position
        fill = reference * (1 - slip)
        proceeds = position['qty'] * fill * (1 - fee) - risk.fixed_exit_fee
        pnl = proceeds - position['cost']
        cash += proceeds
        trades.append(dict(entry_date=position['date'], exit_date=str(date.date()),
                           entry=position['entry'], exit=fill, quantity=position['qty'],
                           pnl=pnl, return_pct=100 * pnl / position['cost'], exit_reason=reason))
        position = None

    for i in range(start, end):
        row, previous = f.iloc[i], f.iloc[i - 1]
        date = f.index[i]
        tradable = row.Volume > 0
        # Indicators for sizing, trend exits and entries always come from yesterday.
        if position is None and i > start and i < end - 1 and signals.iloc[i - 1] and tradable:
            if abs(row.Open - previous.Close) <= risk.max_gap_atr * previous.atr:
                fill = row.Open * (1 + slip)
                distance = strategy.stop_atr * previous.atr
                qty = position_size(cash, fill, distance, previous.volume_reference, risk)
                if qty > 0 and fill > distance:
                    cost = qty * fill * (1 + fee)
                    cash -= cost
                    position = dict(qty=qty, entry=fill, cost=cost, stop=fill - distance,
                                    target=fill + strategy.reward_risk * distance,
                                    date=str(date.date()), bar=i)
        if position is not None and tradable:
            if row.Open <= position['stop']:
                close_position(row.Open, date, 'gap_stop')
            elif i - position['bar'] >= strategy.max_hold or (i > position['bar'] and strategy.trend_exit_ema
                                                             and previous.Close < previous[f'ema{strategy.trend_exit_ema}']):
                # This exit is known at the open, before this session's high/low.
                close_position(row.Open, date, 'time_or_trend')
            elif row.Low <= position['stop']:
                close_position(position['stop'], date, 'stop')
            elif row.High >= position['target']:
                close_position(position['target'], date, 'target')
            elif i == end - 1:
                close_position(row.Close, date, 'segment_end')
        marked = cash if position is None else cash + position['qty'] * row.Close * (1 - slip) * (1 - fee) - risk.fixed_exit_fee
        equity.append(marked)
        dates.append(str(date.date()))
    values = np.array(equity)
    pnl = np.array([t['pnl'] for t in trades])
    positive, negative = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
    years = max((f.index[end - 1] - f.index[start]).days / 365.25, 1 / 365.25)
    total = values[-1] / risk.capital - 1
    benchmark = risk.allocation * ((f.Close.iloc[end - 1] * (1 - slip) * (1 - fee)) /
                                  (f.Open.iloc[start] * (1 + slip) * (1 + fee)) - 1)
    benchmark -= risk.fixed_exit_fee / risk.capital
    metrics = dict(start=str(f.index[start].date()), end=str(f.index[end - 1].date()),
                   trades=len(trades), net_return_pct=100 * total,
                   cagr_pct=100 * ((values[-1] / risk.capital) ** (1 / years) - 1),
                   max_drawdown_pct=100 * float((1 - values / np.maximum.accumulate(values)).max()),
                   profit_factor=positive / negative if negative else None,
                   gross_profit=positive, gross_loss=negative,
                   win_rate_pct=100 * float((pnl > 0).mean()) if len(pnl) else 0.0,
                   expectancy_rupees=float(pnl.mean()) if len(pnl) else 0.0,
                   buy_hold_matched_allocation_pct=100 * benchmark,
                   excess_over_buy_hold_pct=100 * (total - benchmark),
                   unclosed_position=position is not None)
    return dict(metrics=metrics, trades=trades, equity=[dict(date=d, equity=float(v)) for d, v in zip(dates, values[1:])])


def passes(metrics, min_trades, min_pf=1.15, max_drawdown=15.0):
    pf_ok = metrics['profit_factor'] is not None and metrics['profit_factor'] >= min_pf
    if metrics['gross_loss'] == 0 and metrics['gross_profit'] > 0:
        pf_ok = True  # None represents no observed losing trades, never an invented infinity.
    return (metrics['trades'] >= min_trades and metrics['net_return_pct'] > 0 and pf_ok
            and metrics['max_drawdown_pct'] <= max_drawdown and not metrics['unclosed_position'])


def select_strategy(f, risk, min_test_trades=5, institutional_mask=None, strategies=STRATEGIES):
    warmup = 200
    usable = len(f) - warmup
    if usable < 630:
        raise ValueError('Need at least 830 valid daily bars (200 warmup + 630 evaluation)')
    train_end = warmup + int(usable * .5)
    validation_end = warmup + int(usable * .75)
    leaderboard = []
    for strategy in strategies:
        train = simulate(f, strategy, risk, warmup, train_end, institutional_mask)
        validation = simulate(f, strategy, risk, train_end, validation_end, institutional_mask)
        valid = passes(train['metrics'], 3, 1.0) and passes(validation['metrics'], 3)
        score = validation['metrics']['net_return_pct'] - .5 * validation['metrics']['max_drawdown_pct']
        leaderboard.append(dict(strategy=asdict(strategy), train=train, validation=validation,
                                development_passed=bool(valid), validation_score=score))
    eligible = [r for r in leaderboard if r['development_passed']]
    if not eligible:
        return dict(accepted=False, reason='No strategy passed training and validation', development=leaderboard)
    selected = max(eligible, key=lambda r: r['validation_score'])
    strategy = Strategy(**selected['strategy'])
    test = simulate(f, strategy, risk, validation_end, len(f), institutional_mask)
    stress_risk = Risk(**{**asdict(risk), 'slippage_bps': risk.slippage_bps * 2})
    stress = simulate(f, strategy, stress_risk, validation_end, len(f), institutional_mask)
    accepted = passes(test['metrics'], min_test_trades) and passes(stress['metrics'], min_test_trades, 1.0)
    return dict(accepted=bool(accepted), reason='Passed untouched test and doubled-slippage stress' if accepted else 'Selected strategy failed holdout or cost stress',
                strategy=selected['strategy'], train=selected['train'], validation=selected['validation'],
                test=test, stress=stress, development=leaderboard)


def make_plan(symbol, f, selection, ownership, risk):
    strategy = Strategy(**selection['strategy'])
    row = f.iloc[-1]
    slip = risk.slippage_bps / 10000
    distance = strategy.stop_atr * row.atr
    entry = row.Close * (1 + slip)
    quantity = position_size(risk.capital, entry, distance, row.volume_reference, risk)
    active = bool(entry_signal(f, strategy, risk).iloc[-1])
    if distance >= entry:
        quantity = 0
    return dict(symbol=symbol, status='ARMED_NEXT_SESSION' if active and quantity else 'WATCH',
                strategy=asdict(strategy), institutional_signal=ownership,
                price_date=str(f.index[-1].date()), reference_close=float(row.Close),
                average_daily_volume=float(row.volume_reference),
                entry=dict(rule='After a qualifying close, buy next session open; cancel if gap exceeds limit',
                           signal_present=active, max_absolute_gap_rupees=float(risk.max_gap_atr * row.atr),
                           estimated_fill=float(entry)),
                exit=dict(stop_distance_rupees=float(distance), stop_estimate=float(entry-distance),
                          target_estimate=float(entry+distance*strategy.reward_risk),
                          reward_risk=strategy.reward_risk, max_holding_sessions=strategy.max_hold,
                          trend_exit=f'Next open after close below EMA{strategy.trend_exit_ema}' if strategy.trend_exit_ema else 'No moving-average exit',
                          intrabar_assumption='Stop first if both stop and target touched'),
                sizing=dict(reference_capital=risk.capital, quantity_estimate=quantity,
                            risk_fraction=risk.risk_fraction, max_allocation=risk.allocation,
                            max_volume_participation=risk.max_participation,
                            recalculate_at_execution=True),
                risk_model=asdict(risk),
                backtest={part: selection[part]['metrics'] for part in ('train', 'validation', 'test', 'stress')},
                validity='ARMED expires after the next session. WATCH needs a fresh scan after a qualifying close.',
                execution_enabled=False)


# ===========================================================================
# INDIVIDUAL STOCK STRATEGIES
# These are declared research hypotheses, not claims of profitable performance.
# ===========================================================================
import argparse
import json
from datetime import datetime
from pprint import pformat
import time


RULES = {
    'breakout20': Strategy('breakout20', 'breakout', 20, 2.0, 2.5, 30, 1.3),
    'breakout10': Strategy('breakout10', 'breakout', 10, 1.5, 2.5, 20, 1.0),
    'pullback': Strategy('pullback', 'pullback', 20, 2.0, 2.0, 20, .8),
    'reclaim': Strategy('reclaim', 'ema_reclaim', 50, 1.5, 2.5, 30, 1.0),
    'rebound': Strategy('rebound', 'rsi_rebound', 14, 1.5, 2.0, 20, .8, trend_exit_ema=0),
    'wide_breakout': Strategy('wide_breakout', 'breakout', 20, 2.5, 2.0, 30, 1.5),
}

RULE_DESCRIPTIONS = {
    'breakout20': 'Close above prior 20-session high; RSI14 50-80; volume >=1.3x prior 20-session average.',
    'breakout10': 'Close above prior 10-session high; RSI14 50-80; volume >=1.0x prior 20-session average.',
    'pullback': 'Low touches EMA20, bullish close above EMA20 and open; RSI14 40-65; volume >=0.8x prior average.',
    'reclaim': 'Close crosses above EMA50; RSI14 45-65; volume >=1.0x prior average.',
    'rebound': 'RSI14 crosses above 35 and close exceeds previous close; volume >=0.8x prior average; countertrend allowed.',
    'wide_breakout': 'Close above prior 20-session high; RSI14 50-80; volume >=1.5x prior average; wider ATR stop.',
}

# First rule is the default returned by the stock's individual Python function.
# Backtests may select the second rule using development results only.
STOCKS = {
    'ANANDRATHI.NS': ('Anand Rathi Wealth', ('breakout10', 'reclaim')),
    'CENTRALBK.NS': ('Central Bank of India', ('pullback', 'breakout20')),
    'EMCURE.NS': ('Emcure Pharmaceuticals', ('reclaim', 'pullback')),
    'GLAXO.NS': ('GlaxoSmithKline Pharmaceuticals', ('rebound', 'pullback')),
    'HFCL.NS': ('HFCL', ('breakout20', 'breakout10')),
    'OLAELEC.NS': ('Ola Electric Mobility', ('rebound', 'breakout10')),
    'POONAWALLA.NS': ('Poonawalla Fincorp', ('reclaim', 'pullback')),
    'RADICO.NS': ('Radico Khaitan', ('pullback', 'breakout20')),
    'PREMIERENE.NS': ('Premier Energies', ('breakout20', 'reclaim')),
    'ANTELOPUS.NS': ('Antelopus Selan Energy (oil and gas, not solar)', ('wide_breakout', 'reclaim')),
    'IOLCP.NS': ('IOL Chemicals and Pharmaceuticals', ('rebound', 'breakout20')),
    'YESBANK.NS': ('YES Bank', ('reclaim', 'pullback')),
    'SUZLON.NS': ('Suzlon Energy', ('breakout10', 'pullback')),
    'IDEA.NS': ('Vodafone Idea', ('wide_breakout', 'rebound')),
    'PNB.NS': ('Punjab National Bank', ('pullback', 'reclaim')),
    'BANKINDIA.NS': ('Bank of India', ('pullback', 'breakout20')),
    'UNIONBANK.NS': ('Union Bank of India', ('reclaim', 'pullback')),
    'UCOBANK.NS': ('UCO Bank', ('breakout20', 'pullback')),
    'IOB.NS': ('Indian Overseas Bank', ('breakout20', 'pullback')),
    'NHPC.NS': ('NHPC', ('breakout20', 'pullback')),
    'SJVN.NS': ('SJVN', ('breakout20', 'reclaim')),
    'IRFC.NS': ('Indian Railway Finance Corporation', ('pullback', 'breakout20')),
    'IDFCFIRSTB.NS': ('IDFC First Bank', ('reclaim', 'pullback')),
    'GMRAIRPORT.NS': ('GMR Airports', ('breakout10', 'pullback')),
    'RPOWER.NS': ('Reliance Power', ('wide_breakout', 'rebound')),
}
REQUESTED_SYMBOLS = tuple(STOCKS)[:11]
DISCOVERY_SEEDS = tuple(STOCKS)[11:]


def budget_risk(capital=10000.0):
    """Reference costs, not a claim about the user's actual broker tariff."""
    return Risk(capital=capital, risk_fraction=.01, allocation=.40, fixed_exit_fee=15.34)


def stock_signals(symbol, prices, variant=None, risk=None):
    """Return causal technical signals. Current FII/DII is NOT broadcast backwards.

    All non-rebound rules also require close > EMA200 and EMA50 > EMA200.
    A True entry at close t is evaluated for a fill at open t+1, never at close t.
    Stops: entry minus stop_atr*ATR. Target: entry plus reward_risk*stop distance.
    These arrays are hypotheses, not authorization to buy or evidence of profit.
    """
    symbol = symbol_name(symbol)
    variants = STOCKS[symbol][1]
    variant = variant or variants[0]
    if variant not in variants:
        raise ValueError(f'{symbol} supports {variants}, not {variant}')
    rule, risk = RULES[variant], risk or budget_risk()
    f = features(validate_prices(prices))
    result = pd.DataFrame(index=f.index)
    result['enter_next_open'] = entry_signal(f, rule, risk)
    result['exit_next_open'] = (f.Close < f[f'ema{rule.trend_exit_ema}']) if rule.trend_exit_ema else False
    result['atr14'] = f.atr
    result['stop_distance'] = f.atr * rule.stop_atr
    result['target_distance'] = result.stop_distance * rule.reward_risk
    result['max_holding_sessions'] = rule.max_hold
    result.attrs.update(symbol=symbol, rule=asdict(rule), scope='technical_only', execution_enabled=False)
    return result


def strategy_anandrathi(prices, **kwargs):
    """Anand Rathi: 10-day breakout or EMA50 recovery."""
    return stock_signals('ANANDRATHI.NS', prices, **kwargs)

def strategy_centralbk(prices, **kwargs):
    """Central Bank: trend pullback or 20-day breakout."""
    return stock_signals('CENTRALBK.NS', prices, **kwargs)

def strategy_emcure(prices, **kwargs):
    """Emcure: EMA50 recovery or trend pullback; history gate still applies."""
    return stock_signals('EMCURE.NS', prices, **kwargs)

def strategy_glaxo(prices, **kwargs):
    """GLAXO: RSI35 rebound or trend pullback."""
    return stock_signals('GLAXO.NS', prices, **kwargs)

def strategy_hfcl(prices, **kwargs):
    """HFCL: 20-day or faster 10-day breakout."""
    return stock_signals('HFCL.NS', prices, **kwargs)

def strategy_olaelec(prices, **kwargs):
    """Ola Electric: RSI35 rebound or 10-day breakout; history gate still applies."""
    return stock_signals('OLAELEC.NS', prices, **kwargs)

def strategy_poonawalla(prices, **kwargs):
    """Poonawalla: EMA50 recovery or trend pullback."""
    return stock_signals('POONAWALLA.NS', prices, **kwargs)

def strategy_radico(prices, **kwargs):
    """Radico Khaitan: trend pullback or 20-day breakout."""
    return stock_signals('RADICO.NS', prices, **kwargs)

def strategy_premierene(prices, **kwargs):
    """Premier Energies: 20-day breakout or EMA50 recovery."""
    return stock_signals('PREMIERENE.NS', prices, **kwargs)

def strategy_antelopus(prices, **kwargs):
    """Antelopus SELAN Energy: wide-stop breakout or EMA50 recovery."""
    return stock_signals('ANTELOPUS.NS', prices, **kwargs)

def strategy_iolcp(prices, **kwargs):
    """IOL Chemicals: RSI35 rebound or 20-day breakout."""
    return stock_signals('IOLCP.NS', prices, **kwargs)

def strategy_yesbank(prices, **kwargs):
    return stock_signals('YESBANK.NS', prices, **kwargs)

def strategy_suzlon(prices, **kwargs):
    return stock_signals('SUZLON.NS', prices, **kwargs)

def strategy_idea(prices, **kwargs):
    return stock_signals('IDEA.NS', prices, **kwargs)

def strategy_pnb(prices, **kwargs):
    return stock_signals('PNB.NS', prices, **kwargs)

def strategy_bankindia(prices, **kwargs):
    return stock_signals('BANKINDIA.NS', prices, **kwargs)

def strategy_unionbank(prices, **kwargs):
    return stock_signals('UNIONBANK.NS', prices, **kwargs)

def strategy_ucobank(prices, **kwargs):
    return stock_signals('UCOBANK.NS', prices, **kwargs)

def strategy_iob(prices, **kwargs):
    return stock_signals('IOB.NS', prices, **kwargs)

def strategy_nhpc(prices, **kwargs):
    return stock_signals('NHPC.NS', prices, **kwargs)

def strategy_sjvn(prices, **kwargs):
    return stock_signals('SJVN.NS', prices, **kwargs)

def strategy_irfc(prices, **kwargs):
    return stock_signals('IRFC.NS', prices, **kwargs)

def strategy_idfcfirstb(prices, **kwargs):
    return stock_signals('IDFCFIRSTB.NS', prices, **kwargs)

def strategy_gmrairport(prices, **kwargs):
    return stock_signals('GMRAIRPORT.NS', prices, **kwargs)

def strategy_rpower(prices, **kwargs):
    return stock_signals('RPOWER.NS', prices, **kwargs)


def backtest_stock(symbol, prices, risk=None, historical_ownership=None):
    """Choose a rule on training/validation, then evaluate that rule's final period.

    Three development trades and five final-period trades are minimum evidence.
    Reusing previously inspected dates is exploratory, not independent validation.
    """
    symbol, risk = symbol_name(symbol), risk or budget_risk()
    f = features(validate_prices(prices))
    mask = None
    if historical_ownership is not None:
        history = validate_holdings(historical_ownership)
        history = history.loc[history.symbol == symbol]
        if history.empty or history.available_at.isna().any():
            raise ValueError('Original disclosure dates are required for historical institutional gating')
        mask = pd.Series([ownership_signal(history, date, historical=True)['eligible'] for date in f.index], index=f.index)
    result = select_strategy(f, risk, institutional_mask=mask, strategies=tuple(RULES[k] for k in STOCKS[symbol][1]))
    if result['accepted'] and any(result[k]['metrics']['max_drawdown_pct'] > 10 for k in ('train', 'validation', 'test', 'stress')):
        result.update(accepted=False, reason='Exceeded the 10% account drawdown limit')
    result['scope'] = 'price_and_dated_ownership' if mask is not None else 'price_rules_only'
    result['independent_validation'] = False
    return result


def low_price_high_volume(prices, max_price=250, min_volume=1000000, min_turnover=10000000):
    """Small nominal share price is NOT a valuation or quality assessment."""
    p = validate_prices(prices)
    if len(p) < 20:
        return dict(matches=False, reason='Need 20 daily bars')
    last = p.tail(20)
    price, volume, turnover = float(p.Close.iloc[-1]), float(last.Volume.mean()), float((last.Close*last.Volume).mean())
    return dict(matches=bool(price <= max_price and volume >= min_volume and turnover >= min_turnover),
                close=price, average_daily_volume=volume, average_daily_turnover=turnover,
                price_date=str(p.index[-1].date()))


def order_specification(symbol, prices, evaluation, ownership, risk=None, next_open=None):
    """Return an unsubmitted paper-order specification only when all gates pass.

    Prices must end at the immediately preceding completed exchange session.
    The caller must check exchange holidays, quote freshness and existing positions.
    A specification is never an automatic broker order.
    """
    risk = risk or budget_risk()
    if not evaluation.get('accepted') or not ownership.get('eligible') or next_open is None:
        return None
    if not math.isfinite(next_open) or next_open <= 0:
        raise ValueError('Opening price must be finite and positive')
    f = features(validate_prices(prices))
    rule = Strategy(**evaluation['strategy'])
    if not entry_signal(f, rule, risk).iloc[-1] or abs(next_open-f.Close.iloc[-1]) > risk.max_gap_atr*f.atr.iloc[-1]:
        return None
    fill, distance = next_open*(1+risk.slippage_bps/10000), float(f.atr.iloc[-1]*rule.stop_atr)
    quantity = position_size(risk.capital, fill, distance, f.volume_reference.iloc[-1], risk)
    if quantity <= 0 or distance >= fill:
        return None
    return dict(symbol=symbol_name(symbol), quantity=quantity, estimated_fill=fill,
                stop=fill-distance, target=fill+distance*rule.reward_risk,
                max_holding_sessions=rule.max_hold, signal_date=str(f.index[-1].date()),
                status='PAPER_ONLY', submit_automatically=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--list', action='store_true')
    p.add_argument('--snapshot', action='store_true', help='Print the dated research summary embedded in this file, without networking')
    p.add_argument('--run', action='store_true')
    p.add_argument('--discover', action='store_true', help='Measure the 14 additional lower-price/liquid discovery seeds')
    p.add_argument('--symbols', nargs='+')
    p.add_argument('--as-of', default=str(pd.Timestamp.now(tz='Asia/Kolkata').date()))
    p.add_argument('--capital', type=float, default=10000)
    p.add_argument('--account-value', type=float, help='Current total equity; at <=90%% of original capital, allocate nothing')
    p.add_argument('--price-dir', type=Path)
    p.add_argument('--holdings-csv', type=Path)
    p.add_argument('--institutional-backtest', action='store_true')
    p.add_argument('--max-price', type=float, default=250)
    p.add_argument('--min-volume', type=float, default=1000000)
    p.add_argument('--output', type=Path, default=Path('output'))
    args = p.parse_args(argv)
    if args.snapshot:
        print(json.dumps(LAST_RESEARCH_SUMMARY, indent=2))
        return 0
    if args.list or not args.run:
        for symbol, (name, variants) in STOCKS.items():
            print(f'{symbol:17} {name:50} {", ".join(variants)}')
        return 0
    if not all(math.isfinite(x) and x > 0 for x in (args.capital, args.max_price, args.min_volume)):
        p.error('Capital, maximum price and minimum volume must be finite and positive')
    if args.account_value is not None and (not math.isfinite(args.account_value) or args.account_value < 0):
        p.error('Account value must be finite and nonnegative')
    if args.institutional_backtest and not args.holdings_csv:
        p.error('--institutional-backtest needs a holdings CSV with original available_at dates')
    as_of = pd.Timestamp(args.as_of).normalize()
    today = pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None).normalize()
    if pd.isna(as_of) or as_of.tz is not None or as_of > today:
        p.error('--as-of must be a valid, timezone-free date no later than today')
    if as_of != today and not args.holdings_csv:
        p.error('Historical --as-of requires point-in-time ownership CSV data')
    symbols = [symbol_name(s) for s in args.symbols] if args.symbols else list(REQUESTED_SYMBOLS)
    if args.discover:
        symbols.extend(DISCOVERY_SEEDS)
    symbols = list(dict.fromkeys(symbols))
    if any(s not in STOCKS for s in symbols):
        p.error('Unknown symbol; use --list for the explicit per-stock strategy functions')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'prices').mkdir(exist_ok=True)
    supplied = validate_holdings(pd.read_csv(args.holdings_csv)) if args.holdings_csv else None
    risk = budget_risk(args.capital)
    account_stopped = args.account_value is not None and args.account_value <= .9*args.capital
    rows, snapshots = [], []
    throttled = False
    for number, symbol in enumerate(symbols, 1):
        print(f'[{number}/{len(symbols)}] {symbol}', flush=True)
        row = dict(symbol=symbol, name=STOCKS[symbol][0], strategies=[asdict(RULES[k]) for k in STOCKS[symbol][1]],
                   rules=[RULE_DESCRIPTIONS[k] for k in STOCKS[symbol][1]], paper_only=True)
        try:
            prices = load_prices(symbol, as_of-pd.DateOffset(years=7), as_of, args.price_dir)
            now = pd.Timestamp.now(tz='Asia/Kolkata')
            if as_of == today and now.hour*60+now.minute < 16*60:
                prices = prices.loc[prices.index < today]
            if prices.empty or (as_of-prices.index[-1]).days > 5 or prices.Volume.iloc[-1] <= 0:
                raise ValueError('Price data is missing, stale or suspended')
            prices.to_csv(args.output/'prices'/f'{symbol}.csv', index_label='Date')
            row['market'] = low_price_high_volume(prices, args.max_price, args.min_volume)
            row['market']['history_bars'] = len(prices)
            history = None
            try:
                if supplied is not None:
                    history = supplied.loc[supplied.symbol == symbol]
                    if history.empty:
                        raise ValueError('No ownership rows for this symbol')
                elif throttled:
                    raise ValueError('Provider rate-limited; ownership download not attempted')
                else:
                    time.sleep(2)
                    history = public_holdings(symbol, today)
                    snapshots.append(history)
                row['ownership'] = ownership_signal(history, as_of)
            except Exception as exc:
                throttled |= getattr(getattr(exc, 'response', None), 'status_code', None) == 429
                row['ownership'] = dict(eligible=False, reason=str(exc), status='DATA_UNAVAILABLE')
            if args.institutional_backtest and history is None:
                raise ValueError('Historical ownership required but unavailable')
            try:
                row['backtest'] = backtest_stock(symbol, prices, risk, history if args.institutional_backtest else None)
            except ValueError as exc:
                row['backtest'] = dict(accepted=False, reason=str(exc))
            e = row['backtest']
            row['status'] = ('PAPER_WATCH' if e['accepted'] and row['ownership']['eligible'] else 'RESEARCH_ONLY')
            if e['accepted'] and row['ownership']['eligible']:
                plan = make_plan(symbol, features(prices), e, row['ownership'], risk)
                row['plan'] = plan
                row['status'] = 'PAPER_SIGNAL' if plan['entry']['signal_present'] and plan['sizing']['quantity_estimate'] else 'PAPER_WATCH'
            if account_stopped:
                row['status'] = 'ACCOUNT_LOSS_LIMIT'
        except Exception as exc:
            row.update(status='DATA_ERROR', reason=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    # Reserve at most two 40% positions. Rank only by development results.
    ranked = sorted([r for r in rows if r.get('status') == 'PAPER_SIGNAL'],
                    key=lambda r: r['backtest']['validation']['metrics']['net_return_pct']-.5*r['backtest']['validation']['metrics']['max_drawdown_pct'], reverse=True)
    for number, row in enumerate(ranked):
        row['capital_slot_reserved'] = number < 2
        if number >= 2:
            row['status'] = 'PAPER_WATCH_CAPITAL_LIMIT'
            row['plan']['sizing']['quantity_estimate'] = 0
    metadata = dict(as_of=str(as_of.date()), capital=args.capital, account_stopped=account_stopped,
                    risk=asdict(risk), max_positions=2, universe=symbols,
                    discovery_price_ceiling=args.max_price, discovery_min_volume=args.min_volume,
                    ownership_mode='both_new', generated_at=datetime.now().astimezone().isoformat(),
                    scope='historical_institutional_and_price' if args.institutional_backtest else 'price_backtests_current_ownership_screen',
                    note='Exploratory research; repeated historical selection is not independent validation. No live orders. Low price is not low valuation.')
    report = dict(metadata=metadata, stocks=rows)
    text_report = json.dumps(report, indent=2, allow_nan=False)
    (args.output/'research.json').write_text(text_report)
    plain = json.loads(text_report)
    (args.output/'individual_strategies.py').write_text('"""Generated research results; no live execution."""\nRESULTS = '+pformat(plain, sort_dicts=False)+'\n')
    if snapshots:
        pd.concat(snapshots).to_csv(args.output/'ownership_snapshot.csv', index=False)
    lines = ['INDIVIDUAL STOCK STRATEGIES AND RESULTS', '='*40, f'As of {as_of.date()}; capital Rs {args.capital:,.0f}.',
             'All signals are paper research. Institutional eligibility and backtest evidence are separate gates.', '']
    for r in rows:
        lines += [r['symbol']+' — '+r['name'], 'Status: '+r['status']]
        for rule, description in zip(r['strategies'], r['rules']):
            lines.append(f"  {rule['name']}: {description} Stop {rule['stop_atr']:g} ATR; target {rule['reward_risk']:g}R; hold <= {rule['max_hold']} sessions.")
        if 'market' in r:
            m = r['market']
            lines.append(f"  {m['price_date']}: close Rs {m['close']:.2f}, 20-day mean volume {m['average_daily_volume']:,.0f}. Lower-price/high-volume match: {m['matches']}.")
        lines.append('  Ownership: '+r.get('ownership', {}).get('reason', 'unavailable'))
        e = r.get('backtest', {})
        lines.append('  Backtest: '+e.get('reason', r.get('reason', 'unavailable')))
        if 'test' in e:
            for stage in ('train', 'validation', 'test', 'stress'):
                m = e[stage]['metrics']
                lines.append(f"    {stage}: {m['start']} to {m['end']}; {m['trades']} trades; net {m['net_return_pct']:+.2f}%; drawdown {m['max_drawdown_pct']:.2f}%.")
        lines.append('')
    (args.output/'RESEARCH_REPORT.txt').write_text('\n'.join(lines))
    print(f'\nSaved {args.output / "RESEARCH_REPORT.txt"}')
    return 3 if any(r['status']=='DATA_ERROR' or r.get('ownership',{}).get('status')=='DATA_UNAVAILABLE' for r in rows) else 0


# Dated observations only; NEVER used to authorize a new trade.
LAST_RESEARCH_SUMMARY = {'as_of': '2026-10-02',
 'capital': 10000,
 'scope': 'price_backtests_current_ownership_screen',
 'execution_enabled': False,
 'stocks': [{'symbol': 'ANANDRATHI.NS',
             'price_date': '2026-09-30',
             'close': 2120.0,
             'average_daily_volume': 343254,
             'lower_price_high_volume': False,
             'institutional_reversal': True,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'CENTRALBK.NS',
             'price_date': '2026-10-01',
             'close': 30.0,
             'average_daily_volume': 6824016,
             'lower_price_high_volume': True,
             'institutional_reversal': True,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'Selected strategy failed holdout or cost stress',
             'selected_strategy': 'breakout20',
             'final_period_trades': 0,
             'final_period_return_pct': 0.0},
            {'symbol': 'EMCURE.NS',
             'price_date': '2026-09-30',
             'close': 1943.0,
             'average_daily_volume': 208238,
             'lower_price_high_volume': False,
             'institutional_reversal': True,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'Need at least 830 valid daily bars (200 warmup + 630 evaluation)'},
            {'symbol': 'GLAXO.NS',
             'price_date': '2026-09-30',
             'close': 2769.3,
             'average_daily_volume': 75536,
             'lower_price_high_volume': False,
             'institutional_reversal': True,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'HFCL.NS',
             'price_date': '2026-10-01',
             'close': 238.36,
             'average_daily_volume': 6719038,
             'lower_price_high_volume': True,
             'institutional_reversal': True,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'Selected strategy failed holdout or cost stress',
             'selected_strategy': 'breakout20',
             'final_period_trades': 1,
             'final_period_return_pct': 1.5813},
            {'symbol': 'OLAELEC.NS',
             'price_date': '2026-10-01',
             'close': 37.08,
             'average_daily_volume': 95885983,
             'lower_price_high_volume': True,
             'institutional_reversal': True,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'Need at least 830 valid daily bars (200 warmup + 630 evaluation)'},
            {'symbol': 'POONAWALLA.NS',
             'price_date': '2026-10-01',
             'close': 433.4,
             'average_daily_volume': 1220869,
             'lower_price_high_volume': False,
             'institutional_reversal': True,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'RADICO.NS',
             'price_date': '2026-10-01',
             'close': 4237.0,
             'average_daily_volume': 235561,
             'lower_price_high_volume': False,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'PREMIERENE.NS',
             'price_date': '2026-10-01',
             'close': 878.05,
             'average_daily_volume': 1237377,
             'lower_price_high_volume': False,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'Need at least 830 valid daily bars (200 warmup + 630 evaluation)'},
            {'symbol': 'ANTELOPUS.NS',
             'price_date': '2026-09-30',
             'close': 1125.9,
             'average_daily_volume': 3868092,
             'lower_price_high_volume': False,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'IOLCP.NS',
             'price_date': '2026-10-01',
             'close': 209.39,
             'average_daily_volume': 3972279,
             'lower_price_high_volume': True,
             'institutional_reversal': True,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'YESBANK.NS',
             'price_date': '2026-10-01',
             'close': 20.75,
             'average_daily_volume': 78606475,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'SUZLON.NS',
             'price_date': '2026-10-01',
             'close': 38.91,
             'average_daily_volume': 52138452,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'IDEA.NS',
             'price_date': '2026-10-01',
             'close': 12.64,
             'average_daily_volume': 418403815,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'PNB.NS',
             'price_date': '2026-10-01',
             'close': 109.57,
             'average_daily_volume': 12706213,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'BANKINDIA.NS',
             'price_date': '2026-10-01',
             'close': 129.48,
             'average_daily_volume': 4704886,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'UNIONBANK.NS',
             'price_date': '2026-10-01',
             'close': 167.59,
             'average_daily_volume': 7660517,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'UCOBANK.NS',
             'price_date': '2026-10-01',
             'close': 22.93,
             'average_daily_volume': 3434960,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'IOB.NS',
             'price_date': '2026-10-01',
             'close': 30.38,
             'average_daily_volume': 2453138,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'NHPC.NS',
             'price_date': '2026-10-01',
             'close': 71.75,
             'average_daily_volume': 9368283,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'SJVN.NS',
             'price_date': '2026-10-01',
             'close': 57.91,
             'average_daily_volume': 2769701,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'Selected strategy failed holdout or cost stress',
             'selected_strategy': 'reclaim',
             'final_period_trades': 0,
             'final_period_return_pct': 0.0},
            {'symbol': 'IRFC.NS',
             'price_date': '2026-10-01',
             'close': 77.06,
             'average_daily_volume': 9175908,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'IDFCFIRSTB.NS',
             'price_date': '2026-10-01',
             'close': 80.39,
             'average_daily_volume': 21412159,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'GMRAIRPORT.NS',
             'price_date': '2026-10-01',
             'close': 93.16,
             'average_daily_volume': 11087735,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'},
            {'symbol': 'RPOWER.NS',
             'price_date': '2026-10-01',
             'close': 19.41,
             'average_daily_volume': 20031667,
             'lower_price_high_volume': True,
             'institutional_reversal': False,
             'status': 'RESEARCH_ONLY',
             'backtest_reason': 'No strategy passed training and validation'}]}


if __name__ == '__main__':
    raise SystemExit(main())
