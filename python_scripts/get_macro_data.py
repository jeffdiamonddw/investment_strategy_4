import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURATION ---
API_TOKEN = '693327461e9541.04731237'
FRED_KEY = '835b84250468b5de8c18889b86369f7c'
START_DATE = '2005-09-05'
END_DATE = datetime.now().strftime('%Y-%m-%d')
ROTATION_DAYS = 28
SMOOTHING_WINDOW = 21

# Output File
OUTPUT_FILE = 'simulation_data/macro_signals.csv'

def fetch_vix(ticker):
    """Historical EOD endpoint for VIX and VIX3M."""
    url = f"https://eodhd.com/api/eod/{ticker}.INDX?api_token={API_TOKEN}&fmt=json&from={START_DATE}"
    r = requests.get(url)
    if r.status_code == 200:
        df = pd.DataFrame(r.json())
        df['date'] = pd.to_datetime(df['date'])
        return df[['date', 'close']].rename(columns={'close': ticker})
    return pd.DataFrame()

def fetch_ust_yields():
    """Dedicated Treasury API for 10Y and 2Y yields."""
    all_years = []
    current_year = datetime.now().year
    for year in range(2005, current_year + 1):
        url = f"https://eodhd.com/api/ust/yield-rates?api_token={API_TOKEN}&filter[year]={year}&fmt=json"
        r = requests.get(url)
        if r.status_code == 200:
            response_json = r.json()
            # EXTRACT DATA FROM NESTED KEY
            data = response_json.get('data', [])
            
            if isinstance(data, list) and len(data) > 0:
                year_df = pd.DataFrame.from_records(data)
                all_years.append(year_df)
    
    if not all_years: 
        print("No Treasury data found. Check API entitlement.")
        return pd.DataFrame()
        
    df = pd.concat(all_years, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df_pivot = pd.pivot_table(df, index = 'date', values = 'rate', columns = 'tenor')
    if '2Y' in df_pivot.columns and '10Y' in df_pivot.columns:
        df_pivot['y10'] = pd.to_numeric(df_pivot['10Y'], errors='coerce')
        df_pivot['y2'] = pd.to_numeric(df_pivot['2Y'], errors='coerce')
        df_pivot['YIELD_SPREAD'] = df_pivot['y10'] - df_pivot['y2']
        return df_pivot.reset_index()[['date', 'YIELD_SPREAD']]
    return pd.DataFrame()

def fetch_fed_rate():
    """
    Direct access to FRED for the Effective Federal Funds Rate (Daily).
    Series ID: DFF (provides daily history back to 1954).
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "DFF",
        "api_key": FRED_KEY,
        "file_type": "json",
        "observation_start": START_DATE
    }
    print(f"Requesting DFF (Fed Rate) directly from FRED...")
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json().get('observations', [])
        df = pd.DataFrame(data)
        if 'date' in df.columns and 'value' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df.dropna(subset=['value'])[['date', 'value']].rename(columns={'value': 'FED_RATE'})
    return pd.DataFrame()

    
def fetch_reconstructed_hy_spread(treasury_df):
    """
    Reconstructs the HY Spread using HYG ETF and the already fetched Treasury data.
    """
    # Get HYG (High Yield Bond ETF) as proxy for corporate yields
    # &period=d and &order=a ensure we get the full daily history
    hyg_url = f"https://eodhd.com/api/eod/HYG.US?api_token={API_TOKEN}&fmt=json&from={START_DATE}&period=d&order=a"
    r = requests.get(hyg_url)
    
    if r.status_code == 200:
        df_h = pd.DataFrame(r.json())
        df_h['date'] = pd.to_datetime(df_h['date'])
        
        # Merge with the Treasury data provided
        df_merged = df_h.merge(treasury_df[['date', 'YIELD_SPREAD']], on='date', how='inner')
        
        # Proxy Calculation: Invert the price action to create a spread-like stress indicator
        df_merged['HY_SPREAD'] = (1 / df_merged['close']) * 100
        return df_merged[['date', 'HY_SPREAD']]
    return pd.DataFrame()
   



print("Downloading US Treasury Yields (2005-Present)...")
yields = fetch_ust_yields()
hy_spread = fetch_reconstructed_hy_spread(yields)

print("Downloading Credit and Policy Indicators...")
# Indicator slugs: 'real_interest_rate' and 'high_yield_spread' (or equivalent)
fed_rate = fetch_fed_rate()



# 1. DOWNLOAD DATA STREAMS
print("Downloading VIX and VIX3M...")
vix = fetch_vix("VIX")
vix3m = fetch_vix("VIX3M")





# 2. MERGE INTO MASTER DATAFRAME
print("Aligning data and cleaning...")
df_master = vix.merge(vix3m, on='date', how='outer')
df_master = df_master.merge(yields, on='date', how='outer')
df_master = df_master.merge(fed_rate, on='date', how='outer')
df_master = df_master.merge(hy_spread, on='date', how='outer')

df_master = df_master.sort_values('date').ffill().dropna()

# 3. CALCULATE DERIVED SIGNALS
df_master['VIX_RATIO'] = df_master['VIX'] / df_master['VIX3M']

# 4. APPLY 21-DAY SMOOTHING (The Weather Filter)
signals = ['VIX_RATIO', 'YIELD_SPREAD', 'HY_SPREAD', 'FED_RATE']
for sig in signals:
    df_master[f'{sig}_SMOOTH'] = df_master[sig].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

# 5. SAMPLE AT 4-WEEK ROTATION INTERVALS
rotation_dates = []
current = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)

df_master.set_index('date', inplace=True)

while current <= end_dt:
    # Find nearest actual trading day
    idx_pos = df_master.index.get_indexer([current], method='nearest')[0]
    rotation_dates.append(df_master.index[idx_pos])
    current += timedelta(days=ROTATION_DAYS)

# Select smoothed columns at intervals
final_cols = ['date'] + [f'{s}_SMOOTH' for s in signals]
df_weather = df_master.iloc[df_master.index.get_indexer(rotation_dates)].copy().reset_index()
df_weather = df_weather[final_cols].drop_duplicates(subset='date')

# 6. SAVE AND REPORT
df_weather.to_csv(OUTPUT_FILE, index=False)
print(f"\nSUCCESS: Data saved to {OUTPUT_FILE}")
print(df_weather.tail(10))