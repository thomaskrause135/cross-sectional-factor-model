import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

def compute_monthly_ic(factor_df: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """
    Computes the monthly Spearman rank Information Coefficient (IC)
    between factor scores at month t and realized returns at month t+1.
    """
    ic_values = {}
    common_dates = factor_df.index.intersection(forward_returns.index)
    
    for dt in common_dates:
        f = factor_df.loc[dt]
        r = forward_returns.loc[dt]
        valid = f.notna() & r.notna()
        if valid.sum() >= 10:
            corr, _ = spearmanr(f[valid], r[valid])
            ic_values[dt] = corr
            
    return pd.Series(ic_values, name="IC")

def walk_forward_ridge_backtest(
    factors: dict[str, pd.DataFrame],
    train_window: int = 24,
    alpha: float = 50.0,         # Increased default from 10.0 for stability
    top_n: int = 5,
    mode: str = "expanding",     # "rolling" or "expanding"
    cost_bps: float = 40.0       # 40 bps round-trip cost per unit of turnover
) -> pd.DataFrame:
    """
    Runs a walk-forward Ridge regression to combine factors without look-ahead bias.
    Simulates an equal-weighted Long-Short portfolio (Long top_n / Short bottom_n).
    Deducts transaction costs based on monthly portfolio turnover.
    """
    feature_names = [k for k in factors.keys() if k != "forward_return"]
    fwd_ret = factors["forward_return"]
    dates = fwd_ret.dropna(how="all").index
    
    # Build long-format panel: (Date, Ticker) -> features + target
    panel_frames = []
    for name in feature_names:
        s = factors[name].stack()
        s.name = name
        panel_frames.append(s)
    y_series = fwd_ret.stack()
    y_series.name = "target"
    panel = pd.concat(panel_frames + [y_series], axis=1).dropna()
    
    results = []
    prev_longs, prev_shorts = set(), set()
    
    for i in range(train_window, len(dates)):
        pred_date = dates[i]
        
        # Determine training dates based on mode
        if mode == "expanding":
            train_dates = dates[:i]
        else:
            train_dates = dates[i - train_window : i]
            
        train_data = panel.loc[panel.index.get_level_values(0).isin(train_dates)]
        if pred_date not in panel.index.get_level_values(0) or len(train_data) < 50:
            continue
            
        test_data = panel.loc[pred_date]
        if len(test_data) < top_n * 2:
            continue
            
        X_train = train_data[feature_names].values
        y_train = train_data["target"].values
        X_test = test_data[feature_names].values
        
        # Heavy L2 regularization prevents overfitting noisy stock returns
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        preds = pd.Series(model.predict(X_test), index=test_data.index)
        actuals = test_data["target"]
        
        # Rank IC for the combined ML signal
        ic, _ = spearmanr(preds, actuals)
        
        # Long-Short Top-N Spread Return
        ranked = preds.sort_values(ascending=False)
        curr_longs = set(ranked.index[:top_n])
        curr_shorts = set(ranked.index[-top_n:])
        
        long_ret = actuals.loc[list(curr_longs)].mean()
        short_ret = actuals.loc[list(curr_shorts)].mean()
        ls_gross_ret = long_ret - short_ret
        
        # Calculate Turnover: % of the portfolio that changes each month
        if len(prev_longs) > 0:
            # Count how many stocks were replaced
            changes = len(curr_longs - prev_longs) + len(curr_shorts - prev_shorts)
            turnover = changes / (2 * top_n)
        else:
            turnover = 1.0 # Initial entry (100% turnover)
            
        # Deduct costs (e.g. 40 bps = 0.004) proportional to turnover
        cost = turnover * (cost_bps / 10000.0)
        ls_net_ret = ls_gross_ret - cost
        
        results.append({
            "Date": pred_date,
            "IC": ic,
            "Long_Return": long_ret,
            "Short_Return": short_ret,
            "Turnover": turnover,
            "Gross_Return": ls_gross_ret,
            "Net_Return": ls_net_ret,
        })
        
        # Save current holdings for next month's turnover calc
        prev_longs, prev_shorts = curr_longs, curr_shorts
        
    return pd.DataFrame(results).set_index("Date")