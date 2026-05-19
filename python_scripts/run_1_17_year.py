import functools
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

from triple_threat import *
from pareto_navigator import mean_annual_drawdown_integral, annualized_return, pct_change_quantile, worst_annual_return

# Reuse your existing classes and utilities from triple_threat.py
# (Assuming TripleThreatProblem and RobustParallelManager are available)

# --- CONFIGURATION FOR PERTURBATION ---
TICKER_FILE = "strategy/multi_dim_stock_list.csv"
PARETO_INPUT_FILE = 'analysis/perturbations_17_year_conservative_star.csv' # The file you extracted
NUM_PERTURBATIONS = 9000
NOISE_SCALE = 0  # 5% COEFF VAR
NUM_WORKERS = 1
BATCH_REQUIREMENT = 1
TIMEOUT = 180
EVAL_FOLDER = "s3://jdinvestment/perturbation_evaluations_17_year"
HOLDINGS_FOLDER = "s3://jdinvestment/perturbation_holdings_17_year"
PERTURBATION_FOLDER = "s3://jdinvestment/perturbations_17_year"


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




import logging


# Setup basic logging to stdout (which AWS Batch sends to CloudWatch)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TripleThreatMonitor")

def generate_perturbations(df_seeds, n_required, xl, xu, var_cols = VAR_COLS, noise_scale=0.05):
    """
    Generates perturbations where noise is scaled by the parent value (CV).
    
    Args:
        df_seeds: DataFrame of Pareto solutions (the "islands").
        n_required: Total number of evaluations to generate.
        xl, xu: Lower and upper bounds for clipping.
        noise_scale: The desired Coefficient of Variation (std / parent_value).
    """
     
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
    def __init__(self, num_workers, timeout_sec, workhorse_cls, workhorse_args, target_completions):
        self.num_workers = num_workers
        self.timeout_sec = timeout_sec
        self.workhorse_cls = workhorse_cls
        self.workhorse_args = workhorse_args
        self.target_completions = target_completions # Wait for 90/94
        
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


    def force_reset_fleet(self):
        """
            Nuclear reset: Terminates all workers, replaces corrupted queues, 
            and respawns the compute fleet to ensure a clean state for the next batch.
        """
        # 1. Kill the existing workers
        # We use terminate() first to allow for a slightly cleaner OS-level cleanup,
        # then follow up with SIGKILL for any stragglers.
        for p in self.workers:
            try:
                if p.is_alive():
                    p.terminate() 
            except Exception:
                pass

        # Give the OS a moment to reap the processes
        time.sleep(0.1)

        # 2. Hard-Kill and Join
        # Ensures no 'zombie' entries remain in the process table
        for p in self.workers:
            try:
                if p.is_alive():
                    os.kill(p.pid, signal.SIGKILL)
                p.join(timeout=0.1)
            except Exception:
                pass

        # 3. Re-create the communication channels
        # Replacing the queues is the only way to guarantee the internal 
        # locks/pipes aren't in a corrupted state from the terminations.
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()

        # 4. Re-spawn the fleet
        self.workers = []
        for i in range(self.num_workers):
            # We pass the NEW queue references here
            p = mp.Process(
                target=self._worker_loop, 
                args=(
                    self.input_queue, 
                    self.output_queue, 
                    self.workhorse_cls, 
                    self.workhorse_args, 
                    i
                ),
                daemon=True
            )
            p.start()
            self.workers.append(p)
        
        # Reset any generation-specific state tracking
        self.prev_target_pos = None

        
    def run_batch(self, X):
        """
        Submits 94 evaluations and releases once 90 are finished.
        Undone evaluations are returned to be re-queued.
        """
       
        
        pending = range(X.shape[0])
        while len(pending) > 0:
            
            logger.info("{} jobs are pending".format(len(pending)))
            # 2. Push tasks to the fleet
            num_to_submit = min(self.num_workers, len(pending))
            jobs_to_submit = pending[:num_to_submit]
            for idx in jobs_to_submit:
                self.input_queue.put((idx, X[idx]))

        
            # 3. Collection Loop
            completions = 0
            start_time = time.time()
            while completions < self.target_completions and len(pending) > 0:
                elapsed = time.time() - start_time
                time_remaining = self.timeout_sec - elapsed
                
                if completions >= self.target_completions or time_remaining <= 0:
                    break

                try:
                    idx, val, success = self.output_queue.get(timeout=max(0.1, time_remaining))
                    pending = list(set(pending).difference([idx]))
                    completions += 1
                except mp.queues.Empty:
                    break
            self.force_reset_fleet()
 

       

    def cleanup(self):
        for _ in range(self.num_workers):
            self.input_queue.put(STOP_SIGNAL)
        for p in self.workers:
            p.join()



if __name__ == "__main__":
    periods = {
        '17_year': {'train_start_date': pd.to_datetime('Jan 1, 2005'), 'val_start_date': pd.to_datetime('Jan 1, 2008'), 'end_date': pd.to_datetime('Jan 1, 2025')},
    }

    weight_cols = [
        'dollar_ret_1p', 'dollar_ret_6p', 'dollar_ret_13p',
       'dollar_ret_26p', 'avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q',
       'avg_eps_8q', 'threshold', 'beta', 'mom_decay', 'qual_decay',
       'macro_weights_0', 'macro_weights_1', 'macro_weights_2',
       'macro_weights_3'
    ]
    
    
    tickers = list(pd.read_csv(TICKER_FILE).symbol) + ['GIC']
    objective_functions_dict = {
        'worst_annual_return': functools.partial(worst_annual_return, pd.to_datetime('2008-01-01'), pd.to_datetime('2025-01-01'), tickers),
        'drawdown_integral': functools.partial(mean_annual_drawdown_integral, pd.to_datetime('2008-01-01'), pd.to_datetime('2025-01-01'), tickers),
        'annualized_return': functools.partial(annualized_return, pd.to_datetime('2008-01-01'), pd.to_datetime('2025-01-01'), tickers),
        'worst_annual_4wk': functools.partial(pct_change_quantile, pd.to_datetime('2008-01-01'), pd.to_datetime('2025-01-01'), tickers, 1/13),
    }
    objective_sense = {'drawdown_integral': 'min', 'annualized_return': 'max', 'worst_annual_4wk': 'max'}


    problem_args = get_triple_threat_params(
        periods,
        weight_cols, 
        objective_functions_dict, 
        objective_sense,
        momentum_file = "simulation_data/momentum.nc", 
        quality_file = "simulation_data/quality.nc",
        gic_file = "simulation_data/gic_data.nc",
        macro_file = "simulation_data/macro_signals.csv",
        manifold_file = "sim_results/manifold_triple_threat.csv",
        holdings_folder = HOLDINGS_FOLDER,
        eval_folder = EVAL_FOLDER


    )
    xl, xu = problem_args[-2:]

    df_pareto = pd.read_csv(PARETO_INPUT_FILE)
   
    
    # Generate 10,000 samples around Pareto seeds
    X_all = df_pareto[weight_cols].values
    

    manager = HighThroughputBatchManager(num_workers = NUM_WORKERS, timeout_sec =TIMEOUT, workhorse_cls = TripleThreatProblem, workhorse_args = problem_args, target_completions = BATCH_REQUIREMENT)
    
    manager.run_batch(X_all)
    
    

    manager.cleanup()