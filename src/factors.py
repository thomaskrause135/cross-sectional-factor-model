import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sector map for the 43 JSE tickers we track (before data-quality filtering).
# Used by sector-neutral Z-scores to standardize factors *within* a sector
# rather than across the whole index, so commodity beta doesn't dominate.
# Edit these classifications freely if you disagree (e.g. SOL, REM, BVT).
# ---------------------------------------------------------------------------
SECTOR_MAP: dict[str, str] = {
    # Resources, Mining, Energy & Chemicals
    "AGL.JO": "Resources",
    "ANG.JO": "Resources",
    "BHG.JO": "Resources",
    "GFI.JO": "Resources",
    "GLN.JO": "Resources",
    "HAR.JO": "Resources",
    "IMP.JO": "Resources",
    "NPH.JO": "Resources",
    "PAN.JO": "Resources",
    "S32.JO": "Resources",
    "SOL.JO": "Resources",   # Sasol — often grouped with Resources/Energy on JSE
    "SSW.JO": "Resources",
    "VAL.JO": "Resources",
    # Banks, Insurance & Financial Services
    "ABG.JO": "Financials",
    "CPI.JO": "Financials",
    "DSY.JO": "Financials",
    "FSR.JO": "Financials",
    "INL.JO": "Financials",
    "INP.JO": "Financials",
    "NED.JO": "Financials",
    "OMU.JO": "Financials",
    "OUT.JO": "Financials",
    "REM.JO": "Financials",  # Remgro — investment holding
    "RNI.JO": "Financials",  # Reinet — investment holding
    "SBK.JO": "Financials",
    "SLM.JO": "Financials",
    # Tech & Telecoms
    "MTN.JO": "Tech",
    "NPN.JO": "Tech",
    "PRX.JO": "Tech",
    "VOD.JO": "Tech",
    # Consumer, Healthcare & Industrials
    "ANH.JO": "Consumer",
    "APN.JO": "Consumer",
    "BID.JO": "Consumer",
    "BTI.JO": "Consumer",
    "BVT.JO": "Consumer",    # Bidvest — services/industrials
    "CFR.JO": "Consumer",
    "CLS.JO": "Consumer",
    "MRP.JO": "Consumer",
    "PPH.JO": "Consumer",
    "SHP.JO": "Consumer",
    "WHL.JO": "Consumer",
    # Real Estate
    "GRT.JO": "RealEstate",
    "NRP.JO": "RealEstate",
}


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes factor values across ALL stocks at each month-end (mean=0, std=1),
    clipping extreme outliers at +/- 3 standard deviations.
    (Original global version — kept for backward compatibility.)
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    z = df.sub(mean, axis=0).div(std, axis=0)
    return z.clip(-3.0, 3.0)


def cross_sectional_zscore_by_group(
    df: pd.DataFrame,
    sector_map: dict[str, str] = SECTOR_MAP,
) -> pd.DataFrame:
    """
    Standardizes factor values *within each sector* at each month-end
    (mean=0, std=1 inside the sector), clipping outliers at +/- 3 std.

    Why: A global Z-score lets a gold rally push every miner into the top
    decile at once, turning stock selection into a commodity-beta bet.
    Grouping by sector forces the model to rank best-miner vs worst-miner
    and best-bank vs worst-bank instead.

    Any ticker in df.columns but missing from sector_map is grouped under
    "Other" so nothing is silently dropped.
    """
    # Build sector -> list of tickers, only for columns actually present
    sectors: dict[str, list[str]] = {}
    for ticker in df.columns:
        sector = sector_map.get(ticker, "Other")
        sectors.setdefault(sector, []).append(ticker)

    z_scored_parts: list[pd.DataFrame] = []
    for sector, tickers in sectors.items():
        sub = df[tickers]
        if len(tickers) == 1:
            # Z-score is undefined with one name; set to 0 (neutral) so the
            # model still sees a valid row rather than a NaN.
            z = pd.DataFrame(0.0, index=df.index, columns=tickers)
        else:
            mean = sub.mean(axis=1)
            std = sub.std(axis=1)
            # Guard against degenerate rows (std == 0) — dividing by NaN
            # leaves the result NaN, which the model will correctly drop.
            safe_std = std.replace(0.0, np.nan)
            z = sub.sub(mean, axis=0).div(safe_std, axis=0)
        z_scored_parts.append(z)

    z = pd.concat(z_scored_parts, axis=1)
    return z.clip(-3.0, 3.0)


def compute_monthly_factors(
    daily_prices: pd.DataFrame,
    neutralize: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Computes three classic quantitative equity factors resampled to month-end:
      1. Momentum (12-1 Month): 12-month return skipping the most recent month.
      2. Short-Term Reversal (1-Month): Negative of prior 1-month return.
      3. Low Volatility: Negative of 60-day annualized realized volatility.
    Also returns forward 1-month returns (target variable) aligned properly.

    Parameters
    ----------
    daily_prices : DataFrame of daily adjusted close prices (columns = tickers)
    neutralize   : if True, Z-score factors *within sector* instead of globally.
                   Reduces commodity-beta leakage (see cross_sectional_zscore_by_group).
                   Defaults to False so existing notebooks behave identically.
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

    # Pick the Z-score function based on the neutralize flag
    zscore = cross_sectional_zscore_by_group if neutralize else cross_sectional_zscore

    return {
        "momentum_12_1": zscore(mom_12_1),
        "reversal_1m": zscore(rev_1m),
        "low_volatility": zscore(low_vol),
        "forward_return": forward_returns,
    }