
import awswrangler as wr
import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def calculate_crowding_distance(F):
    """
    Manual implementation of crowding distance.
    Matches pymoo logic: handles boundary points and normalization.
    """
    n_points, n_obj = F.shape
    if n_points <= 2:
        return np.full(n_points, np.inf)
    
    dist = np.zeros(n_points)
    for i in range(n_obj):
        # Sort indices by current objective
        idx = np.argsort(F[:, i])
        
        # Set boundary points to infinity (standard NSGA-II diversity)
        dist[idx[0]] = dist[idx[-1]] = np.inf
        
        obj_range = F[idx[-1], i] - F[idx[0], i]
        # Avoid division by zero if all values in this objective are identical
        if obj_range == 0: 
            continue
            
        for j in range(1, n_points - 1):
            dist[idx[j]] += (F[idx[j+1], i] - F[idx[j-1], i]) / obj_range
            
    return dist

def get_ranked_subset(df, objective_cols, senses, n_required):
    # 1. Validation & Data Prep
    # Ensure objectives are float64 for the C++ backend
    F = df[objective_cols].values.copy().astype(float)
    
    # 0.6.1.6 requirement: Minimize all. Flip max objectives.
    for i, sense in enumerate(senses):
        if sense.lower() == 'max':
            F[:, i] = -F[:, i]

    # 2. Non-Dominated Sorting using the 'efficient_non_dominated_sort' key
    # This matches the specific key listed in your Exception dict_keys
    nds = NonDominatedSorting(method="efficient_non_dominated_sort")
    
    # In 0.6.1.6, nds.do(F) returns a list of arrays (the fronts)
    fronts = nds.do(F)
    
    selected_indices = []
    ranks = np.full(len(df), -1, dtype=int)

    # 3. Hierarchy Extraction
    for k, front in enumerate(fronts):
        # Assign rank to every member of the front (front is a numpy array)
        ranks[front] = k
            
        current_total = len(selected_indices)
        if current_total >= n_required:
            continue # Complete rank assignment for the rest of the 50k points
            
        remaining_needed = n_required - current_total
        
        if len(front) <= remaining_needed:
            # Take the whole rank
            selected_indices.extend(front)
        else:
            # Sub-select from this front using Crowding Distance
            front_f = F[front]
            cd = calculate_crowding_distance(front_f)
            
            # Sort by CD (descending: largest gaps first)
            sub_sort_idx = np.argsort(cd)[::-1]
            chosen_indices = front[sub_sort_idx[:remaining_needed]]
            selected_indices.extend(chosen_indices)
            # Break early if we only care about the subset, but here we 
            # let the loop finish to ensure the 'rank' column is full.

    # 4. Assembly
    # We create a copy to avoid SettingWithCopy warnings
    df_out = df.copy()
    df_out['rank'] = ranks
    
    # Filter for the n_required solutions we manually picked via NDS+CD
    result_df = df_out.iloc[selected_indices].copy()
    
    return result_df.sort_values('rank').reset_index(drop=True)


if __name__ == "__main__":

    database_name = table_name = "robust_nav_medians_4"
    # Read the table metadata and data using the Glue catalo

    df = wr.s3.read_parquet_table(
        table=table_name,
        database=database_name
    )

    #df.loc[:, 'drawdown'] = df.loc[:, ['f1_2008', 'f2_2020', 'f3_2022']].sum(axis = 1)
    #senses = ['min', 'max', 'max']
    #obj_columns = ['drawdown', 'f4_terminal', 'f5_annual_worst']

    obj_columns = ['boom', 'crash']
    senses = ['max', 'max']
    df_ranked = get_ranked_subset(df, obj_columns, senses, 500)

    df_ranked.to_csv('analysis/top_ranked_pareto_nav_median.csv')