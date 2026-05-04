import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr


# Silence the internal yfinance Pandas4Warning
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow is deprecated.*")

# --- CONFIGURATION ---
 # Replace 'YOUR_API_KEY' with your actual EODHD API key
API_KEY = '693327461e9541.04731237' 
TICKER_FILE = 'strategy/multi_dim_stock_list.csv'
MOMENTUM_PATH = 'simulation_data/momentum.nc'
QUALITY_PATH = 'simulation_data/quality.nc'
PIN_DATE = '2005-09-05'
DAYS_INTERVAL = 28

MOM_BANDS = ['price_end', 'dollar_ret_1p', 'dollar_ret_6p', 'dollar_ret_13p', 'dollar_ret_26p']
QUAL_BANDS = ['avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q', 'avg_eps_8q']

LEVERAGE_MAP = {
    'SH': -1.0, 'SDS': -2.0, 'SPXU': -3.0, 'PSQ': -1.0, 'QID': -2.0, 
    'SQQQ': -3.0, 'BTAL': -1.0, 'SVXY': 0.5, 'CTA': -1.0, 'DBMF': -1.0, 
    'KMLM': -1.0, 'PFIX': -1.0, 'CYA': -1.0, 'RSBT': -1.0, 'FMF': -1.0
}

# --- 1. DIAGNOSTIC ALIGNMENT CHECK ---
def create_gic_dataarray(df: pd.DataFrame) -> xr.DataArray:
    """
    Constructs an xarray DataArray from a GIC dataframe.
    
    Args:
        df: DataFrame with datetime index and 'gic' column (annual rate).
    """
    # 1. Calculate the periodic growth factor (28 days)
    # Using 28/365 as the fraction of the year
    df = df.copy()
    df['growth_factor'] = (1 + .01 * df['gic']) ** (28 / 365)
    
    # 2. Define the windows for trailing returns
    windows = [1, 6, 13, 26]
    
    # 3. Initialize the DataArray with dimensions
    bands = ['price_end'] + [f'dollar_ret_{w}p' for w in windows]
    symbols = ['GIC']
    dates = np.array(df.index)
    
    # Initialize with NaNs
    da = xr.DataArray(
        np.nan,
        coords={'band': bands, 'symbol': symbols, 'date': dates},
        dims=('band', 'symbol', 'date')
    )
    
    # 4. Fill 'price_end' with 1.0
    da.loc[dict(band='price_end')] = 1.0
    
    # 5. Calculate trailing dollar returns
    # The return for a window k is the cumulative product of growth factors 
    # over the last k periods.
    for w in windows:
        # We use a rolling window product. 
        # min_periods=w ensures we only get values when we have enough data.
        # This calculates the value of 1 dollar invested w periods ago.
        trailing_returns = (
            df['growth_factor']
            .rolling(window=w)
            .apply(lambda x: x.prod(), raw=True)
        )
        da.loc[dict(band=f'dollar_ret_{w}p')] = trailing_returns.values
        
    return da - 1


def get_boc_simulation_data_2026(start_date='2005-01-01'):
    """
    Pulls Bank Rate and 1-Year GIC directly from the BoC Valet API.
    Bypasses group-level 404 errors by requesting specific series.
    """
    # Vectors: V80691310 (Bank Rate), V80691339 (1-Year GIC)
    series_ids = "V80691310,V80691339"
    url = f"https://www.bankofcanada.ca/valet/observations/{series_ids}/csv?start_date={start_date}"
    
    try:
        print(f"Requesting series {series_ids} from Bank of Canada...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # BoC CSVs include metadata at the top. We split to find the data table.
        lines = response.text.splitlines()
        data_start = next(i for i, line in enumerate(lines) if line.startswith('"date"'))
        
        # Read data and clean column names
        df = pd.read_csv(io.StringIO("\n".join(lines[data_start:])))
        df.columns = [c.strip('"') for c in df.columns]
        df = df.rename(columns={'date': 'REF_DATE'})
        
        # Melt to match your simulation's expected [REF_DATE, VECTOR, VALUE] format
        df_melted = df.melt(id_vars=['REF_DATE'], var_name='VECTOR', value_name='VALUE')
        
        # Standardize Vector IDs to lowercase for your existing logic
        df_melted['VECTOR'] = df_melted['VECTOR'].str.lower()
        df_melted['REF_DATE'] = pd.to_datetime(df_melted['REF_DATE'])
        
        print(f"Success! Data retrieved up to {df_melted['REF_DATE'].max().date()}")
        return df_melted.sort_values(['VECTOR', 'REF_DATE']).reset_index(drop=True)

    except Exception as e:
        print(f"BoC Direct Retrieval Failed: {e}")
        return pd.DataFrame()







def run_strict_alignment_check(da_mom, da_qual, csv_tickers):
    """Ensures 1:1 correspondence between CSV, Momentum, and Quality."""
    print("\n" + "="*60)
    print("--- SCRATCH REBUILD AUDIT ---")
    print(f"Target Tickers (CSV): {len(csv_tickers)}")
    print(f"Momentum Symbols:     {len(da_mom.symbol)}")
    print(f"Quality Symbols:      {len(da_qual.symbol)}")
    
    mom_zero_pct = (da_mom == 0).sum().values / da_mom.size * 100
    qual_zero_pct = (da_qual == 0).sum().values / da_qual.size * 100
    
    print(f"Momentum Fill Rate: {100 - mom_zero_pct:.2f}%")
    print(f"Quality Fill Rate:  {100 - qual_zero_pct:.2f}%")
    
    if np.array_equal(da_mom.symbol.values, da_qual.symbol.values):
        print("[PASS] Ticker alignment successful.")
    else:
        print("[FAIL] Ticker mismatch detected!")

    if np.array_equal(da_mom.date.values, da_qual.date.values):
        print(f"[PASS] Date alignment successful ({len(da_mom.date)} dates).")
    else:
        print("[FAIL] Date mismatch detected!")
    print("="*60 + "\n")

# --- 2. CORE MOMENTUM BUILDER ---

import requests
import io

def build_momentum(tickers, target_dates):
    """Downloads price data and calculates returns for all CSV tickers using EODHD."""
    print(f"Downloading Momentum data for {len(tickers)} tickers from EODHD...")
    full_data = np.zeros((len(MOM_BANDS), len(tickers), len(target_dates)))
    
   
    
    for j, ticker in enumerate(tickers):
        try:
            # EODHD End-of-Day API URL
            # US tickers use the .US exchange suffix; adjust if your CSV includes exchanges
            url = f'https://eodhd.com/api/eod/{ticker}.US?api_token={API_KEY}&fmt=csv&from={PIN_DATE}'
            
            response = requests.get(url)
            if response.status_code != 200:
                print(f"    ! Failed to fetch {ticker} (Status: {response.status_code})")
                continue
            
            # Read CSV response
            df = pd.read_csv(io.StringIO(response.text), index_col='Date', parse_dates=True)
            
            # EODHD 'Adjusted_close' handles splits and dividends automatically
            if 'Adjusted_close' not in df.columns:
                print(f"    ! Adjusted data missing for {ticker}")
                continue
                
            # Alignment and cleaning using your existing logic
            series = df['Adjusted_close'].reindex(target_dates, method='ffill').ffill().bfill()

            # --- SURGICAL REPAIR FOR SPLIT ANOMALIES ---
            # Even with quality data, clipping remains a safety protocol
            ratios = series / series.shift(1)
            series = series.mask((ratios > 5.0) | (ratios < 0.2)).ffill()
            
            # Explicit cast to float for NumPy assignment
            full_data[0, j, :] = series.values.astype(float)
            
            for i, window in enumerate([1, 6, 13, 26], start=1):
                full_data[i, j, :] = series.pct_change(periods=window).fillna(0).values

            print('got momentum data for {} {}/{}'.format(ticker, j+1, len(tickers)))
                
        except Exception as e:
            print(f"    ! Error processing {ticker}: {e}")
            continue

    return xr.DataArray(full_data, coords={'band': MOM_BANDS, 'symbol': tickers, 'date': target_dates}, dims=['band', 'symbol', 'date'])

# --- 3. INTEGRATED QUALITY BUILDER ---

import requests
import io

def build_quality(tickers, target_dates, da_mom):
    """
    Recreates Quality bands with logic-based branching:
    - Short ETFs: Use Price Proxy.
    - Stocks: Fetch actual EPS from EODHD.
    - Long ETFs: Fetch holdings and calculate weighted average EPS.
    """
    results = []
    print(f"Recreating Quality data for {len(tickers)} tickers...")
    
    # Load metadata for branching logic
    ticker_meta = pd.read_csv(TICKER_FILE).set_index('symbol')
    

    for j, ticker in enumerate(tickers):

        #try:
            
        
        # 1. Determine Logic Path from Metadata
        asset_type = ticker_meta.loc[ticker, 'asset_type'].upper()
        direction = ticker_meta.loc[ticker, 'direction'].upper()
        
        data = np.zeros((4, len(target_dates)))

        # PATH A: Short ETFs (Price Proxy Only)
        if asset_type == 'ETF' and direction == 'SHORT':
            leverage = LEVERAGE_MAP.get(ticker, -1.0)
            p_etf = da_mom.sel(symbol=ticker, band='price_end')
            p_mkt = da_mom.sel(band='price_end').mean(dim='symbol')
            raw_proxy = (p_etf / p_mkt) * leverage
            
            for i, window in enumerate([1, 2, 4, 8]):
                data[i] = pd.Series(raw_proxy.values).rolling(window=window, min_periods=1).mean().values

        # PATH B: Stocks (Direct EPS)
        elif asset_type == 'STOCK':
            url = f'https://eodhd.com/api/fundamentals/{ticker}.US?api_token={API_KEY}'
            resp = requests.get(url).json()
            # Extract quarterly earnings and align to target_dates
            earnings = resp.get('Earnings', {}).get('History', {})
            if earnings:
                eps_series = pd.DataFrame.from_dict(earnings, orient='index')
                eps_series.index = pd.to_datetime(eps_series['date'])
                aligned_eps = eps_series['epsActual'].reindex(target_dates, method='ffill').fillna(0)
                for i, window in enumerate([1, 2, 4, 8]):
                    data[i] = aligned_eps.rolling(window=window, min_periods=1).mean().values
            else:
                raise ValueError("No EPS data found")

        # PATH C: Long ETFs (Weighted Average of Holdings EPS)
        elif asset_type == 'ETF' and direction == 'LONG':
            holdings_file = f'./holdings/{ticker}_holdings.csv'
            
            if os.path.exists(holdings_file):
                # Read local CSV. Based on your snippet, we use 'Code' and 'Assets_%'
                df_holdings = pd.read_csv(holdings_file)
                
                weighted_eps_sum = pd.Series(0.0, index=target_dates)
                total_w = 0.0
                
                for _, row in df_holdings.iterrows():
                    # Extract the symbol (CDE, FCX, etc.) and append .US if needed
                    h_raw_code = str(row['Code']).strip()
                    h_code = f"{h_raw_code}.US" if "." not in h_raw_code else h_raw_code
                    
                    # Use 'Assets_%' column, converting to a decimal (e.g., 6.16 -> 0.0616)
                    h_weight = float(row['Assets_%']) / 100.0
                    
                    # Fetch individual holding EPS from EODHD
                    h_url = f'https://eodhd.com/api/fundamentals/{h_code}?api_token={API_KEY}'
                    h_resp = requests.get(h_url).json()
                    h_hist = h_resp.get('Earnings', {}).get('History', {})
                    
                    if h_hist:
                        h_df = pd.DataFrame.from_dict(h_hist, orient='index')
                        h_df.index = pd.to_datetime(h_df['date'])
                        h_series = h_df['epsActual'].reindex(target_dates, method='ffill').fillna(0)
                        weighted_eps_sum += (h_series * h_weight)
                        total_w += h_weight
                
                if total_w > 0:
                    final_eps = weighted_eps_sum / total_w
                    for i, window in enumerate([1, 2, 4, 8]):
                        data[i] = final_eps.rolling(window=window, min_periods=1).mean().values
                    success = True

        print('got quality data for {} {}/{}'.format(ticker, j+1, len(tickers)))

        

        da = xr.DataArray(data[:, np.newaxis, :], 
                          coords={'band': QUAL_BANDS, 'symbol': [ticker], 'date': target_dates}, 
                          dims=['band', 'symbol', 'date'])
        results.append(da)

    return xr.concat(results, dim='symbol')


def get_gic_data():
    df_gic = get_boc_simulation_data_2026()

    # 2. Pivot data (v80691310 = Bank Rate, v80691339 = 1-year GIC)
    pivot_df = df_gic.pivot(index='REF_DATE', columns='VECTOR', values='VALUE')
    pivot_df = pivot_df.rename(columns={
        'v80691310': 'Bank_Rate',
        'v80691339': 'GIC_1_Year'
    })

    # 3. Create 4-week intervals starting March 31, 2005
    price_data = xr.open_dataarray('simulation_data/momentum.nc')
    date_range = np.array(price_data.date)
    processed_df = pivot_df.reindex(pivot_df.index.union(date_range)).ffill().reindex(date_range)

    # 4. Updated Approximation for Cashable GIC
    # Formula: Midpoint between Bank Rate and 1-Year GIC minus 0.35%
    # This results in a current value of 2.00% (based on 2.25% Bank and 2.45% GIC)
    processed_df['gic'] = ((processed_df['Bank_Rate'] + processed_df['GIC_1_Year']) / 2) - 0.35

    da = create_gic_dataarray(processed_df)
    return da
    
# --- 4. MAIN ENTRY POINT ---

def main():
    os.makedirs('simulation_data', exist_ok=True)
    # for path in [MOMENTUM_PATH, QUALITY_PATH]:
    #     if os.path.exists(path):
    #         os.remove(path)

    csv_tickers = sorted(pd.read_csv(TICKER_FILE)['symbol'].unique().tolist())
    
    # Modern Pandas handling to avoid Timestamp.utcnow warnings
    now_utc = pd.Timestamp.now(tz='UTC').tz_localize(None) 
    target_dates = pd.date_range(start=PIN_DATE, end=now_utc, freq=f'{DAYS_INTERVAL}D')

    if os.path.isfile(MOMENTUM_PATH):
        da_mom = xr.open_dataarray(MOMENTUM_PATH)
        new_target_dates = sorted(list(set(target_dates).difference(np.array(da_mom.date))))
        if len(new_target_dates) > 0:
            da_new_dates = build_momentum(np.array(da_mom.symbol), new_target_dates)
            da_mom = xr.concat([da_mom, da_new_dates], dim = 'date')
        new_tickers = sorted(list(set(csv_tickers).difference(np.array(da_mom.symbol))))
        if len(new_tickers) > 0:
            da_new_tickers = build_momentum(new_tickers, target_dates)
            da_mom = xr.concat([da_mom, da_new_tickers], dim = 'symbol')
        if len(new_target_dates) > 0 or len(new_tickers) > 0:
            da_mom.to_netcdf(MOMENTUM_PATH)
    else:   
        da_mom = build_momentum(csv_tickers, target_dates)
        da_mom.to_netcdf(MOMENTUM_PATH)

    
    
    
    if os.path.isfile(QUALITY_PATH):
        da_qual = xr.open_dataarray(QUALITY_PATH)
        new_target_dates = sorted(list(set(target_dates).difference(np.array(da_qual.date))))
        if len(new_target_dates) > 0:
            da_new_dates = build_quality(np.array(da_qual.symbol), new_target_dates, da_mom)
            da_qual = xr.concat([da_qual, da_new_dates], dim = 'date')
        new_tickers = sorted(list(set(csv_tickers).difference(np.array(da_qual.symbol))))
        if len(new_tickers) > 0:
            da_new_tickers = build_quality(new_tickers, target_dates, da_mom)
            da_qual = xr.concat([da_qual, da_new_tickers], dim = 'symbol')
        if len(new_target_dates) > 0 or len(new_tickers) > 0:
            da_qual.to_netcdf(QUALITY_PATH)
    else:
        da_qual = build_quality(csv_tickers, target_dates, da_mom)
        da_qual.to_netcdf(QUALITY_PATH)

    run_strict_alignment_check(da_mom, da_qual, csv_tickers)

   
   

    da_gic = get_gic_data()
    da_gic.to_netcdf('simulation_data/gic_data_1.nc')



    
    print("Full rebuild complete and verified.")

if __name__ == "__main__":
    main()