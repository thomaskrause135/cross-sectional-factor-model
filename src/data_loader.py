import pandas as pd
import yfinance as yf

# Full FTSE/JSE Top 40 Constituents (Yahoo Finance .JO tickers)
JSE_TOP_40_TICKERS = [
    # Basic Resources & Mining
    "AGL.JO",  # Anglo American
    "ANG.JO",  # AngloGold Ashanti
    "BHG.JO",  # BHP Group
    "GFI.JO",  # Gold Fields
    "GLN.JO",  # Glencore
    "HAR.JO",  # Harmony Gold Mining
    "IMP.JO",  # Impala Platinum
    "NPH.JO",  # Northam Platinum
    "PAN.JO",  # Pan African Resources
    "S32.JO",  # South32
    "SOL.JO",  # Sasol
    "SSW.JO",  # Sibanye-Stillwater
    "VAL.JO",  # Valterra Platinum (formerly Amplats)

    # Banks, Insurance & Financial Services
    "ABG.JO",  # Absa Group
    "CPI.JO",  # Capitec Bank
    "DSY.JO",  # Discovery
    "FSR.JO",  # FirstRand
    "INL.JO",  # Investec Ltd
    "INP.JO",  # Investec plc
    "NED.JO",  # Nedbank Group
    "OMU.JO",  # Old Mutual
    "OUT.JO",  # OUTsurance Group
    "REM.JO",  # Remgro
    "RNI.JO",  # Reinet Investments
    "SBK.JO",  # Standard Bank
    "SLM.JO",  # Sanlam

    # Technology & Telecommunications
    "MTN.JO",  # MTN Group
    "NPN.JO",  # Naspers
    "PRX.JO",  # Prosus
    "VOD.JO",  # Vodacom Group

    # Consumer, Retail, Healthcare & Industrials
    "ANH.JO",  # Anheuser-Busch InBev
    "APN.JO",  # Aspen Pharmacare
    "BID.JO",  # Bidcorp
    "BTI.JO",  # British American Tobacco
    "BVT.JO",  # Bidvest Group
    "CFR.JO",  # Richemont
    "CLS.JO",  # Clicks Group
    "MRP.JO",  # Mr Price Group
    "PPH.JO",  # Pepkor Holdings
    "SHP.JO",  # Shoprite Holdings
    "WHL.JO",  # Woolworths Holdings

    # Real Estate
    "GRT.JO",  # Growthpoint Properties
    "NRP.JO",  # NEPI Rockcastle
]

def fetch_jse_prices(
    tickers: list[str] = JSE_TOP_40_TICKERS,
    start_date: str = "2018-01-01",
    end_date: str = "2026-11-01"
) -> pd.DataFrame:
    """
    Downloads daily adjusted close prices for JSE equities from Yahoo Finance.
    Drops tickers with excessive missing history to prevent alignment gaps.
    """
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    prices = raw["Close"]
    
    # Forward-fill up to 3 days for occasional halted/holiday ticks, then keep stocks with >95% data
    prices = prices.ffill(limit=3)
    valid_cols = prices.columns[prices.isna().mean() < 0.05]
    prices = prices[valid_cols].dropna()
    
    return prices
