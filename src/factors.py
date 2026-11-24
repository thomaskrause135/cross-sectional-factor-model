import numpy as np
import pandas as pd

def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes factor values across all stocks at each month-end (mean=0, std=1),
    clipping extreme outliers at +/- 3 standard deviations.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    z = df.sub(mean, axis=0).div(std, axis=0)
    return z.clip(-3.0, 3.0)

def compute_monthly_factors(daily_prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Computes three classic quantitative equity factors resampled to month-end:
      1. Momentum (12-1 Month): 12-month return skipping the most recent month.
      2. Short-Term Reversal (1-Month): Negative of prior 1-month return.
      3. Low Volatility: Negative of 60-day annualized realized volatility.
    Also returns forward 1-month returns (target variable) aligned properly.
    """
    # Resample daily prices to month-end ('ME' in pandas >= 2.2)
    monthly_prices = daily_prices.resample("ME").last()
    
    # Target: Next month's return (shifted back by 1 so row t holds return from t to t+1)
    forward_returns = monthly_prices.pct_change().shift(-1)
    
    # Factor 1: 12-1 Momentum (skips the most recent month to avoid short-term pullback noise)
    mom_12_1 = (monthly_prices.shift(1) / monthly_prices.shift(12)) - 1.0
    
    # Factor 2: 1-Month Reversal (stocks that dropped sharply over 1 month often bounce back)
    rev_1m = -monthly_prices.pct_change(1)
    
    # Factor 3: Low Volatility (calmer stocks tend to earn higher risk-adjusted returns)
    daily_returns = daily_prices.pct_change()
    realized_vol_60d = daily_returns.rolling(window=60).std() * np.sqrt(252)
    low_vol = -realized_vol_60d.resample("ME").last()
    
    return {
        "momentum_12_1": cross_sectional_zscore(mom_12_1),
        "reversal_1m": cross_sectional_zscore(rev_1m),
        "low_volatility": cross_sectional_zscore(low_vol),
        "forward_return": forward_returns,
    }
