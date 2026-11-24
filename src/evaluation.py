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
    alpha: float = 10.0,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Runs a rolling walk-forward Ridge regression to combine factors without
    look-ahead bias, then simulates an equal-weighted Long-Short portfolio
    (Long top_n predicted stocks, Short bottom_n predicted stocks).
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
    
    # Walk forward month-by-month starting after initial 24-month training window
    for i in range(train_window, len(dates)):
        pred_date = dates[i]
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
        
        # Heavy L2 regularization (alpha=10) prevents overfitting noisy stock returns
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        preds = pd.Series(model.predict(X_test), index=test_data.index)
        actuals = test_data["target"]
        
        # Rank IC for the combined ML signal
        ic, _ = spearmanr(preds, actuals)
        
        # Long-Short Top-N Spread Return (before transaction costs)
        ranked = preds.sort_values(ascending=False)
        long_ret = actuals.loc[ranked.index[:top_n]].mean()
        short_ret = actuals.loc[ranked.index[-top_n:]].mean()
        ls_ret = long_ret - short_ret
        
        results.append({
            "Date": pred_date,
            "IC": ic,
            "Long_Return": long_ret,
            "Short_Return": short_ret,
            "Long_Short_Return": ls_ret,
        })
        
    return pd.DataFrame(results).set_index("Date")
