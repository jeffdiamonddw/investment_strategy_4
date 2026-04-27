import multiprocessing as mp
import numpy as np
import time
from pymoo.core.problem import Problem

import os
import numpy as np
import pandas as pd
import xarray as xr
import boto3
import time
import pickle
import fsspec
from pymoo.core.problem import Problem
import multiprocessing as mp

from manifold_dry_run_parallel import *

# Reuse your existing classes and utilities from manifold_dry_run_parallel.py
# (Assuming TripleThreatProblem and RobustParallelManager are available)

# --- CONFIGURATION FOR PERTURBATION ---
PARETO_INPUT_FILE = "analysis/top_ranked.csv" # The file you extracted
NUM_PERTURBATIONS = 10000
NOISE_SCALE = 0.05  # 5% COEFF VAR
NUM_WORKERS = 94
BATCH_REQUIREMENT = 90
TIMEOUT = 180
EVAL_FILE = "s3://jdinvestment/perturbation_evaluations_4"
HOLDINGS_FILE = "s3://jdinvestment/perturbation_holdings_4"
PERTURBATION_FOLDER = "s3://jdinvestment/perturbations_4"


# Indices: 0-7: PCA, 8: Threshold, 9: Beta, 10-11: Decay, 12-15: Macro Weights
XL_DEFAULT = np.array([
    -2, -2, -2, -2,  # Mom PCA
    -2, -2, -2, -2,  # Qual PCA
    -2.0,            # Threshold (Index 8: expanded from 0.1)
    0.5,             # Beta (Index 9)
    -1, -1,          # Decays
    -1, -1, -1, -1   # Macro Weights
])

XU_DEFAULT = np.array([
    2, 2, 2, 2,      # Mom PCA
    2, 2, 2, 2,      # Qual PCA
    2.0,             # Threshold (Index 8: expanded from 0.9)
    15.0,            # Beta (Index 9: expanded from 2.0)
    1, 1,            # Decays
    1, 1, 1, 1       # Macro Weights
])
VAR_COLS = [

            'dollar_ret_1p',
            'dollar_ret_6p',
            'dollar_ret_13p',
            'dollar_ret_26p',
            'avg_eps_1q',
            'avg_eps_2q',
            'avg_eps_4q',
            'avg_eps_8q', 
            'threshold', 
            'beta',
            'mom_decay',
            'qual_decay',
            'macro_weights_0',
            'macro_weights_1',
            'macro_weights_2',
            'macro_weights_3'
]


import numpy as np
import pandas as pd

def generate_perturbations(df_seeds, n_required, xl, xu, noise_scale=0.05):
    """
    Generates perturbations where noise is scaled by the parent value (CV).
    
    Args:
        df_seeds: DataFrame of Pareto solutions (the "islands").
        n_required: Total number of evaluations to generate.
        xl, xu: Lower and upper bounds for clipping.
        noise_scale: The desired Coefficient of Variation (std / parent_value).
    """
    var_cols = VAR_COLS 
    seeds = df_seeds[var_cols].values
    parent_ids = df_seeds['sim_id'].values 
    
    n_seeds = len(seeds)
    perturbations_per_seed = int(np.ceil(n_required / n_seeds))
    
    all_x = []
    all_rows = []

    for i, seed in enumerate(seeds):
        # Calculate standard deviation as a percentage of the parent value
        # We use np.abs(seed) to ensure a positive scale for the normal distribution
        # We add a tiny epsilon (1e-8) to prevent zero-noise if a parameter is exactly 0
        local_std = noise_scale * (np.abs(seed) + 1e-8)
        
        # Generate the noise cluster using the local CV-based scale
        noise = np.random.normal(0, local_std, (perturbations_per_seed, len(seed)))
        perturbed_set = seed + noise
        
        # Physical bounds clipping (ensure we don't drift outside the -2 to 2 or 0.1 to 2.0 range)
        perturbed_set = np.clip(perturbed_set, xl, xu)
        
        current_parent = parent_ids[i]
        
        for x_numeric in perturbed_set:
            # Unique sim_id for the perturbation
            new_sim_id = abs(hash(tuple(x_numeric))) % (10**10)
            
            # Construct metadata row
            row_data = {col: val for col, val in zip(var_cols, x_numeric)}
            row_data['sim_id'] = new_sim_id
            row_data['parent_id'] = current_parent
            
            all_x.append(x_numeric)
            all_rows.append(row_data)
            
    # Truncate to exact requirement and return
    X_perturbed = np.array(all_x)[:n_required]
    df_meta = pd.DataFrame(all_rows)[:n_required]
    
    return X_perturbed, df_meta

STOP_SIGNAL = "STOP"

class HighThroughputBatchManager(Problem):
    def __init__(self, num_workers, timeout_sec, workhorse_cls, workhorse_args, target_completion):
        self.num_workers = num_workers
        self.timeout_sec = timeout_sec
        self.workhorse_cls = workhorse_cls
        self.workhorse_args = workhorse_args
        self.target_completion = target_completion # Wait for 90/94
        
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.workers = []
        self._spawn_workers()

        # Decision space matches your 10-variable structure
        super().__init__(n_var=10, n_obj=5, n_constr=0, 
                        xl=XL_DEFAULT, 
                        xu=XU_DEFAULT)

    def _spawn_workers(self):
        for i in range(self.num_workers):
            p = mp.Process(
                target=self._worker_loop, 
                args=(self.input_queue, self.output_queue, self.workhorse_cls, self.workhorse_args, i),
                daemon=True
            )
            p.start()
            self.workers.append(p)

    @staticmethod
    def _worker_loop(input_queue, output_queue, workhorse_cls, workhorse_args, worker_id):
        # Local logic initialization to ensure isolation
        local_engine = workhorse_cls(*workhorse_args)
        import boto3
        worker_session = boto3.Session()
        
        while True:
            task = input_queue.get()
            if task == STOP_SIGNAL:
                break
            
            idx, x_vector = task
            try:
                out_dict = {}
                local_engine._evaluate(x_vector, out_dict, worker_session)
                output_queue.put((idx, out_dict["F"], True))
            except Exception as e:
                output_queue.put((idx, str(e), False))

    def run_batch(self, X):
        """
        Submits 94 evaluations and releases once 90 are finished.
        Undone evaluations are returned to be re-queued.
        """
        n_tasks = X.shape[0]
        results = {} 
        completed_indices = set()
        
        # 1. Enqueue the full batch
        for idx in range(n_tasks):
            self.input_queue.put((idx, X[idx]))

        # 2. Collect until target_completion is reached
        start_time = time.time()
        while len(completed_indices) < self.target_completion:
            try:
                # Use timeout to keep the loop alive and check for total hangs
                idx, val, success = self.output_queue.get(timeout=self.timeout_sec)
                
                if success:
                    results[idx] = val
                    completed_indices.add(idx)
                else:
                    # Robustness: Re-enqueue if the worker explicitly failed
                    self.input_queue.put((idx, X[idx]))
                    
            except mp.queues.Empty:
                print(f"Batch timeout reached at {len(completed_indices)}/{self.target_completion}")
                break

        # 3. Identify and return undone indices
        all_indices = set(range(n_tasks))
        undone_indices = list(all_indices - completed_indices)
        
        print(f"Batch Complete: {len(completed_indices)} finished, {len(undone_indices)} returned to queue.")
        return results, undone_indices

    def cleanup(self):
        for _ in range(self.num_workers):
            self.input_queue.put(STOP_SIGNAL)
        for p in self.workers:
            p.join()



if __name__ == "__main__":
    print("--- DRY RUN START ---")
    da_mom = xr.open_dataset("simulation_data/momentum.nc").to_array().squeeze()
    da_qual = xr.open_dataset("simulation_data/quality.nc").to_array().squeeze()
    _data_features = xr.concat([da_mom, da_qual], dim='band')
    da_mom_gic = xr.open_dataarray("simulation_data/gic_data.nc")
    da_qual_gic = get_gic_eps(da_mom_gic)
    data_features = xr.concat([_data_features, xr.concat([da_mom_gic, da_qual_gic], dim='band')], dim='symbol').drop_sel(band = 'price_end')
    
    df_price = da_mom.sel(band='price_end').to_pandas()
    
    df_macro = pd.read_csv("simulation_data/macro_signals.csv", index_col=0)
    df_macro.index = pd.to_datetime(df_macro.index)
    
    mapping = {'VIX_RATIO_SMOOTH': 'vix_z', 'YIELD_SPREAD_SMOOTH': 'yield_spread_z', 'HY_SPREAD_SMOOTH': 'hy_spread_z', 'FED_RATE_SMOOTH': 'fed_z'}
    for csv_col, engine_key in mapping.items():
        if csv_col in df_macro.columns:
            rolling_mean = df_macro[csv_col].rolling(window=252, min_periods=1).mean()
            rolling_std = df_macro[csv_col].rolling(window=252, min_periods=1).std()
            df_macro[engine_key] = (df_macro[csv_col] - rolling_mean) / rolling_std.replace(0, 1)

    
    
    dates = sorted(list(set(df_macro.index).intersection(df_price.columns)))
    df_macro = df_macro.loc[dates, [col for col in df_macro.columns if col.endswith('z')]]
    
    params = {
        'principal': [327000, 60000, 21000], 'max_frac': .05, 'feature_horizon_weeks': 104,
        'min_price': 5, 'trade_fee': 7, 'objective_sensitivity': 0.144, 'obj_threshold': 0,
        'start_date': pd.to_datetime('Jan 1, 2005'), 'end_date': pd.Timestamp.now()
    }
    
    training_periods = {
        'boom': {'train_start_date': pd.to_datetime('Jan 1, 2018'), 'end_date': pd.to_datetime('Jan 1, 2025')},
        'crash': {'train_start_date': pd.to_datetime('Nov 1, 2005'), 'end_date': pd.to_datetime('Nov 1, 2012')}
    }

    print("Initializing problem kits...")
    df_man = pd.read_csv("sim_results/manifold_triple_threat.csv")
    mom_cols = ['dollar_ret_1p', 'dollar_ret_6p', 'dollar_ret_13p', 'dollar_ret_26p']
    qual_cols = ['avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q', 'avg_eps_8q']
    
    df_elite = df_man.nlargest(int(len(df_man) * 0.10), 'f4_terminal')
    m_kit = build_kit(df_elite, mom_cols)
    q_kit = build_kit(df_elite, qual_cols)

    df_ranked = pd.read_csv(PARETO_INPUT_FILE)
    df_pareto = df_ranked.loc[df_ranked['rank']==0]
    
    # Generate 10,000 samples around Pareto seeds
    X_all, df_perturb = generate_perturbations(df_pareto,   NUM_PERTURBATIONS, XL_DEFAULT, XU_DEFAULT, NOISE_SCALE)
    df_perturb.to_csv("{}/perturbations.csv".format(PERTURBATION_FOLDER))
    
    
    problem_args = (m_kit, q_kit, df_macro, data_features, df_price, params, training_periods, HOLDINGS_FILE, EVAL_FILE)
    manager = HighThroughputBatchManager(num_workers = NUM_WORKERS, timeout_sec =TIMEOUT, workhorse_cls = TripleThreatProblem, workhorse_args = problem_args, target_completion = BATCH_REQUIREMENT)
    
    all_results = []
    task_pool = list(enumerate(X_all)) # List of (original_idx, x_vector)
    
    print("Starting high-throughput evaluation...")
    
    while len(task_pool) > 0:
        # Take up to 94 tasks from the pool
        current_batch = task_pool[:NUM_WORKERS]
        task_pool = task_pool[NUM_WORKERS:]
        
        # Separate indices for the manager
        batch_indices = [t[0] for t in current_batch]
        batch_X = np.array([t[1] for t in current_batch])
        
        # Run the gate
        results_dict, undone_local_indices = manager.run_batch(batch_X)
        
        # Map local batch indices back to original task_pool indices
        for local_idx in undone_local_indices:
            # Put the actual task back at the front of the pool
            task_pool.insert(0, current_batch[local_idx])
            
        # Store successful results
        for res in results_dict.values():
            all_results.append(res)
            
        print(f"Progress: {len(all_results)}/{NUM_PERTURBATIONS} total evaluations recorded.")

    manager.cleanup()