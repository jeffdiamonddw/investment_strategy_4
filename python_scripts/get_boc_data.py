import numpy as np
import pandas as pd
import requests
import io
import xarray as xr


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
da.to_netcdf('simulation_data/gic_data_1.nc')
