"""Synthetic fixtures verify mechanics; never used as stock performance evidence."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('stock_strategies', Path(__file__).parents[1]/'stock_strategies.py')
import sys
code = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = code
spec.loader.exec_module(code)


def bars(n=900):
    rng = np.random.default_rng(42)
    close = 100*np.exp(np.cumsum(rng.normal(.0004, .015, n)))
    opened = np.r_[close[0], close[:-1]]
    return pd.DataFrame(dict(Open=opened, High=np.maximum(opened, close)*1.01,
                             Low=np.minimum(opened, close)*.99, Close=close, Volume=1000000),
                        index=pd.bdate_range('2021-01-01', periods=n))


def execution_bars():
    f = code.features(bars(220))
    for c in ('Open', 'High', 'Low', 'Close'):
        f[c] = 100.0
    f['High'], f['Low'] = 101., 99.
    f['ema200'], f['ema50'], f['ema20'] = 80., 90., 95.
    f['rsi'], f['atr'] = 60., 2.
    f['high20'], f['volume_reference'], f['turnover'] = 99., 100000., 100000000.
    return f


def test_every_stock_has_individual_callable():
    data = bars(230)
    for symbol in code.STOCKS:
        fn = getattr(code, 'strategy_'+symbol[:-3].lower())
        signals = fn(data)
        assert signals.index.equals(data.index)
        assert signals.attrs['symbol'] == symbol
        assert signals.attrs['execution_enabled'] is False
        assert set(['enter_next_open','exit_next_open','stop_distance','target_distance']).issubset(signals)


def test_signals_do_not_see_future_prices():
    p = bars()
    before = code.features(p)
    p.loc[p.index[600]:, ['Open','High','Low','Close']] *= 3
    after = code.features(p)
    for rule in code.RULES.values():
        pd.testing.assert_series_equal(code.entry_signal(before,rule,code.Risk()).iloc[:600],
                                       code.entry_signal(after,rule,code.Risk()).iloc[:600])


def test_next_open_and_pessimistic_stop():
    f = execution_bars()
    f.loc[f.index[202], ['High','Low']] = [130,80]
    r = code.simulate(f,code.RULES['breakout20'],code.Risk(fee_bps=0,slippage_bps=0),201,204)
    assert r['trades'][0]['entry_date'] == str(f.index[202].date())
    assert r['trades'][0]['exit_reason'] == 'stop'
    assert r['trades'][0]['exit'] == 96


def test_gap_stop_and_fixed_cost():
    f = execution_bars()
    f.loc[f.index[203],['Open','High','Low','Close']] = [80,82,78,81]
    risk = code.Risk(capital=10000,risk_fraction=.01,allocation=.4,fee_bps=0,slippage_bps=0,fixed_exit_fee=15.34)
    r = code.simulate(f,code.RULES['breakout20'],risk,201,205)
    t = r['trades'][0]
    assert t['exit_reason'] == 'gap_stop'
    assert t['exit'] == 80
    assert t['pnl'] == pytest.approx(t['quantity']*(80-100)-15.34)


def test_sizing_covers_fixed_fee_and_integer_shares():
    risk = code.budget_risk()
    q = code.position_size(10000,100,4,1000000,risk)
    assert isinstance(q,int)
    assert q*(4+100*(2*risk.fee_bps+risk.slippage_bps)/10000)+15.34 <= 100
    assert q*100*1.0025 <= 4000


def test_ownership_not_known_before_publication():
    h = code.validate_holdings(pd.DataFrame(dict(symbol=['HFCL']*3,
        quarter_end=['2025-12-31','2026-03-31','2026-06-30'],
        available_at=['2026-01-20','2026-04-20','2026-07-20'],
        fii_pct=[2,1,2],dii_pct=[3,2,3],source=['fixture']*3)))
    assert not code.ownership_signal(h,pd.Timestamp('2026-07-19'),historical=True)['eligible']
    assert code.ownership_signal(h,pd.Timestamp('2026-07-20'),historical=True)['eligible']
    h['available_at'] = pd.NaT
    h['observed_at'] = pd.Timestamp('2026-09-27')
    assert not code.ownership_signal(h,pd.Timestamp('2026-09-27'),historical=True)['eligible']


def test_low_price_alone_does_not_pass_liquidity_screen():
    p = bars(25)
    p['Volume'] = 500
    assert not code.low_price_high_volume(p)['matches']
    p['Volume'] = 2000000
    assert code.low_price_high_volume(p)['matches']


def test_short_history_is_never_claimed_qualified():
    with pytest.raises(ValueError,match='830'):
        code.backtest_stock('PREMIERENE',bars(300))


def test_no_order_without_evidence_or_ownership():
    assert code.order_specification('HFCL',bars(),{'accepted':False},{'eligible':True},next_open=100) is None
    assert code.order_specification('HFCL',bars(),{'accepted':True},{'eligible':False},next_open=100) is None


def test_holdout_does_not_choose_runner_up(monkeypatch):
    calls=[]
    def fake(f,s,r,start,end,institutional_mask=None):
        calls.append((s.name,end))
        testing=end==len(f)
        value=4 if s.name=='breakout20' else 2
        return dict(metrics=dict(trades=10,net_return_pct=-1 if testing else value,
            max_drawdown_pct=1,profit_factor=.5 if testing else 2,gross_loss=10,gross_profit=20,
            unclosed_position=False),trades=[],equity=[])
    monkeypatch.setattr(code,'simulate',fake)
    r=code.backtest_stock('HFCL',bars())
    assert not r['accepted']
    assert all(name=='breakout20' for name,end in calls if end==900)
