import pandas as pd
import requests
import io
import zipfile

def get_rates_by_label():
    """
    Downloads Table 10-10-0122-01, reads the data CSV, 
    and filters by the specific text labels in the 'Rates' column.
    """
    url = "https://www150.statcan.gc.ca/n1/tbl/csv/10100122-eng.zip"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # The exact strings you identified in the 'Rates' column
    target_rates = [
        'Bank rate', 
        'Chartered bank - Guaranteed Investment Certificates: 1 year'
    ]
    
   
    print("Downloading bulk ZIP from Statistics Canada...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Identify the main CSV (the one without 'metadata' in the title)
        csv_name = [n for n in z.namelist() if n.endswith('.csv') and 'metadata' not in n][0]
        with z.open(csv_name) as f:
            # low_memory=False helps with larger tables
            df = pd.read_csv(f, low_memory=False)
    
    # Filter rows based on the text labels in the 'Rates' column
    # We also grab the VECTOR column so your existing scripts can still use v80691310/v80691339
    df_filtered = df[df['Rates'].isin(target_rates)].copy()
    
    # Select and format core columns for your simulation
    output_df = df_filtered[['REF_DATE', 'Rates', 'VALUE']].copy()
    output_df['REF_DATE'] = pd.to_datetime(output_df['REF_DATE'])
    
    # Sort chronologically for your regime-switching models
    output_df = output_df.sort_values(['Rates', 'REF_DATE']).reset_index(drop=True).rename(columns = {'REF_DATE': 'date', 'VALUE': 'value', 'Rates': 'rate'})
    output_df.loc[:, 'rate'] = output_df.rate.map({'Chartered bank - Guaranteed Investment Certificates: 1 year': 'gic_1_year', 'Bank rate': 'bank_rate'})
    output_df = output_df.loc[output_df.value.notnull()]
    
    print(f"Extraction complete: {len(output_df)} rows found.")
    return output_df
       

  
# To use:
df = get_rates_by_label()
zzz=1
# df_interest.to_csv('stats_can_data.csv', index=False)