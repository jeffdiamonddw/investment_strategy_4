"""
Triple Threat Strategy: Dry Run Version
1. MINIMAL POPULATION: Set to 3 to verify pipeline integrity.
2. MINIMAL GENERATIONS: Set to 2 to test the ask/tell loop transitions.
3. OUTPUT VISIBLE: SuppressOutput is disabled to monitor engine prints.
"""

import numpy as np

# --- 1. DRY RUN CONSTANTS ---
ANNUAL_RISK_PERCENTILE = 1/13
NUM_WORKERS = 94
N_OFFSPRINGS = 188
TIMEOUT = 180
POP_SIZE = 180   # Minimal for testing
GEN_COUNT =  600# Minimal for testing
EVAL_FILE = "s3://jdinvestment/new_evaluations_9"
HOLDINGS_FILE = "s3://jdinvestment/new_holdings_history_9"
CHECKPOINT_URI = "s3://jdinvestment/checkpoints/checkpoint_740.pkl"


# Shared Search Space Configuration
VAR_COUNT = 16
OBJ_COUNT = 5

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



import os
import boto3 
import signal


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
import psutil  # You may need to add this to your Docker image/environment

# Setup basic logging to stdout (which AWS Batch sends to CloudWatch)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TripleThreatMonitor")



import numpy as np
import os
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
# Pymoo imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population

# Internal Logic Imports
from regime_switch_3 import RegimeSwitchingProblem
from explore_cluster import get_gic_eps

import awswrangler as wr
import fsspec
import pickle


import resource
import platform
import contextlib


import multiprocessing as mp
import numpy as np
import time
from pymoo.core.problem import Problem



# --- 2. UTILITIES ---

import multiprocessing as mp
import numpy as np
import time
from pymoo.core.problem import Problem

# Sentinel to signal workers to shut down
STOP_SIGNAL = "STOP"


# --- ADD THIS TO SATISFY PICKLE LOAD ---
class DummyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=5, n_constr=0, xl=0, xu=1)
    def _evaluate(self, x, out, *args, **kwargs):
        pass 
# ---------------------------------------

class RobustParallelManager(Problem):
    def __init__(self, num_workers, timeout_sec, workhorse_cls, workhorse_args, xl=XL_DEFAULT, xu=XU_DEFAULT):
        # 1. Store local attributes first
        self.num_workers = num_workers
        self.timeout_sec = timeout_sec
        self.workhorse_cls = workhorse_cls
        self.workhorse_args = workhorse_args
        self.n_gen = 0
        self.n_obj = 5
        self.recycle_interval = 10 
        
        # 2. Setup internal communication
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.workers = []
        self._spawn_workers()

        # 3. SINGLE Pymoo Initialization (Defining the Contract)
        # This replaces BOTH previous super() calls with the correct bounds
        super().__init__(n_var=VAR_COUNT, n_obj=OBJ_COUNT, n_constr=0, xl=xl, xu=xu)

    def _spawn_workers(self):
        """Spins up persistent worker processes."""
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
        """The execution loop running inside each child process."""
        # Initialize the heavy logic locally inside the process
        # This ensures DataFrames are not shared across memory spaces
        local_engine = workhorse_cls(*workhorse_args)
        worker_session = boto3.Session()
        while True:
            task = input_queue.get()
            if task == STOP_SIGNAL:
                break
            
            idx, x_vector = task
            try:
                out_dict = {}
                # Call the existing _evaluate method from your original class
                local_engine._evaluate(x_vector, out_dict, session = worker_session)
                output_queue.put((idx, out_dict["F"], True))
            except Exception as e:
                output_queue.put((idx, str(e), False))

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Manager's evaluation logic with an Absolute Generation Deadline.
        Logs generation health and performance metrics to CloudWatch.
        """
        n_individuals = X.shape[0] 
        results_list = [None] * n_individuals
    
        # DYNAMIC TARGET:
        # In production (94), this is ~89-90. 
        # In your local test (2), this ensures it waits for both.
        target_completions = POP_SIZE
    
   
        
        # 1. Start Timing and System Health Snapshot
        gen_start_time = time.time()
        initial_mem = psutil.virtual_memory().percent
        cpu_load = psutil.cpu_percent()

        # 2. Push all 94 tasks to the fleet
        for idx in range(n_individuals):
            self.input_queue.put((idx, X[idx]))

        received_count = 0
       
        
        # 3. Collection Loop
        for _ in range(n_individuals):
            elapsed = time.time() - gen_start_time
            time_remaining = self.timeout_sec - elapsed
            
            if received_count >= target_completions or time_remaining <= 0:
                break

            try:
                idx, val, success = self.output_queue.get(timeout=max(0.1, time_remaining))
                results_list[idx] = val if success else [1e9] * self.n_obj
                received_count += 1
            except mp.queues.Empty:
                break

        # 4. Calculate Critical Performance Metrics
        total_gen_time = time.time() - gen_start_time
        final_mem = psutil.virtual_memory().percent
        
        # 5. LOG TO CLOUDWATCH
        # Structured to be easily searchable/filterable in the AWS Console
        logger.info(f"--- GENERATION {self.n_gen} MONITORING ---")
        logger.info(f"STATUS: {'SUCCESS' if received_count >= POP_SIZE else 'UNDERPERFORMING'}")
        logger.info(f"THROUGHPUT: {received_count}/{n_individuals} workers finished")
        logger.info(f"GEN_TIME: {total_gen_time:.2f}s (Threshold reached at {received_count} evals)")
        logger.info(f"AVG_WORKER_SPEED: {total_gen_time/max(1, received_count):.2f}s per eval")
        logger.info(f"SYSTEM_CPU: {cpu_load}% | SYSTEM_RAM: {final_mem}% (Δ {final_mem - initial_mem:.1f}%)")
        logger.info(f"----------------------------------------")

        # 6. Penalty Padding
        for i in range(n_individuals):
            if results_list[i] is None:
                results_list[i] = [1e9] * self.n_obj

        # 7. The Nuclear Reset
        # We do this AFTER logging so we don't include spawn time in the Gen performance metrics
        self.force_reset_fleet()

        # 8. Finalize output for Pymoo
        out["F"] = np.array(results_list)
        self.n_gen += 1


    def force_reset_fleet(self):
        # 1. Kill everyone immediately
        for p in self.workers:
            try:
                os.kill(p.pid, signal.SIGKILL)
                p.join(timeout=0.1)
            except:
                pass
                
        # 2. Re-create the communication channels
        # This prevents the 'ppoll' deadlock by providing fresh file descriptors
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()

        # 3. Re-spawn using the original "Fat" args
        self.workers = []
        for i in range(self.num_workers):
            p = mp.Process(
                target=self._worker_loop, 
                args=(self.input_queue, self.output_queue, self.workhorse_cls, self.workhorse_args, i),
                daemon=True
            )
            p.start()
            self.workers.append(p)

    def cleanup(self):
        """Properly shuts down the worker processes."""
        for _ in range(self.num_workers):
            self.input_queue.put(STOP_SIGNAL)
        for p in self.workers:
            p.join()

    def __getstate__(self):
        """
        Exclude non-pickleable objects (Queues and Processes) 
        from the checkpoint.
        """
        # Create a copy of the object's state to avoid modifying the live object
        state = self.__dict__.copy()
        
        # Replace non-pickleable objects with placeholders
        state['input_queue'] = None
        state['output_queue'] = None
        state['workers'] = []
        
        return state

    def __setstate__(self, state):
        """
        Restore the state from a checkpoint. 
        Note: The actual Queues and Workers are re-initialized in the main() resume logic.
        """
        self.__dict__.update(state)


    def recycle_workers(self):
        print(f"--- Gen {self.n_gen}: Full Fleet Refresh & Pipe Flush ---")
        # 1. Force kill existing processes
        for p in self.workers:
            try:
                os.kill(p.pid, signal.SIGKILL)
                p.join(timeout=0.1)
            except:
                pass
        
        # 2. Flush the Pipes
        for q in [self.input_queue, self.output_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    break

        # 3. Re-spawn workers
        self.workers = []
        for i in range(self.num_workers):
            p = mp.Process(
                target=self._worker_loop, 
                args=(self.input_queue, self.output_queue, self.workhorse_cls, self.workhorse_args, i),
                daemon=True
            )
            p.start()
            self.workers.append(p)




def limit_to_1_vcpu():
    # Get the list of available logical cores
    cores = list(os.sched_getaffinity(0))
    if len(cores) > 1:
        # Restrict the process to only the first available core
        os.sched_setaffinity(0, {cores[0]})
        print(f"Limiting execution to 1 vCPU (Core: {cores[0]})")
    else:
        print(f"System already has only 1 vCPU available.")


def limit_to_2_vcpus():
    # Get the list of available logical cores
    cores = list(os.sched_getaffinity(0))
    if len(cores) > 2:
        # Restrict the process to only the first two available cores
        os.sched_setaffinity(0, {cores[0], cores[1]})
        print(f"Limiting execution to 2 vCPUs (Cores: {cores[0]}, {cores[1]})")
    else:
        print(f"System already has {len(cores)} or fewer vCPUs.")

@contextlib.contextmanager
def monitor_peak_memory():
    """
    Context manager to measure the peak Resident Set Size (RSS) 
    of the current process and its children.
    """
    try:
        yield
    finally:
        # Get peak memory usage
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_bytes = usage.ru_maxrss
        
        # On macOS, ru_maxrss is in bytes. 
        # On Linux, it is in kilobytes.
        if platform.system() != 'Darwin':
            peak_bytes *= 1024
            
        peak_mb = peak_bytes / (1024 * 1024)
        print(f"\n--- Memory Report ---")
        print(f"Peak Resident Set Size: {peak_mb:.2f} MB")
        print(f"---------------------")


def save_checkpoint(algorithm, path):
    """
    Saves the algorithm state to Local or S3 using fsspec.
    """
    # Use 'wb' for binary write; fsspec handles the protocol (s3 vs local)
    with fsspec.open(path, 'wb') as f:
        pickle.dump(algorithm, f)
    print(f"--- [CHECKPOINT] State saved to {path} ---")

def load_checkpoint(path):
    """
    Loads the algorithm state if it exists.
    """
    # Check existence via fsspec
    fs, fs_path = fsspec.core.url_to_fs(path)
    if fs.exists(fs_path):
        with fsspec.open(path, 'rb') as f:
            return pickle.load(f)
    return None




def save_result_agnostic(df, path):
    """
    Saves a single evaluation result (row) to a destination.
    
    Local Logic: 
        Appends the row to a single CSV file.
    S3 Logic: 
        Writes a unique CSV to the directory using the sim_id as the filename.
    """
    if path.startswith("s3://"):
        # Ensure we have a sim_id to name the file
        if 'sim_id' in df.columns and not df.empty:
            # We take the first sim_id (assuming one eval per call)
            sim_id = df['sim_id'].iloc[0]
            
            # Clean path to ensure it's treated as a directory
            base_dir = path if path.endswith('/') else path + '/'
            s3_path = f"{base_dir}sim_{sim_id}.csv"
            
            # Use awswrangler to write the individual file
            wr.s3.to_csv(df=df, path=s3_path, index=False)
        else:
            raise ValueError("S3 destination requires 'sim_id' in DataFrame for unique naming.")
            
    else:
        # Local Logic: Standard file append
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        # Append to the file (write header only if file doesn't exist)
        file_exists = os.path.isfile(path)
        df.to_csv(path, mode='a', index=False, header=not file_exists)



def build_kit(df, cols):
    s = StandardScaler().fit(df[cols])
    p = PCA(n_components=4).fit(s.transform(df[cols]))
    return {'scaler': s, 'pca': p, 'columns': cols}

class TripleThreatProblem(ElementwiseProblem):
   
    def __init__(self, m_kit, q_kit, df_macro, data_features, df_price, params, training_periods, holdings_file, xl=XL_DEFAULT, xu=XU_DEFAULT):
        # Store these so they exist for the __getstate__ / __setstate__ logic
        self.m_kit = m_kit
        self.q_kit = q_kit
        self.df_macro = df_macro
        self.data_features = data_features
        self.df_price = df_price
        self.params = params
        self.training_periods = training_periods
        self.holdings_file = holdings_file
        
        # Initialize the base engine
        self.base = RegimeSwitchingProblem(m_kit, q_kit, df_macro, data_features, None, 
                                        df_price, params, training_periods, holdings_file)
        
        super().__init__(n_var=VAR_COUNT, n_obj=OBJ_COUNT, n_constr=0, xl=xl, xu=xu)


    def __getstate__(self):
        """
        Exclude massive dataframes/xarrays from the pickle file.
        This keeps the checkpoint file tiny (KBs instead of GBs).
        """
        state = self.__dict__.copy()
        # Remove the heavy hitters
        keys_to_drop = ['df_macro', 'data_features', 'df_price', 'base']
        for key in keys_to_drop:
            state[key] = None
        return state

    def __setstate__(self, state):
        """Called when unpickling; we will re-inject data in the main loop."""
        self.__dict__.update(state)
        
    def _evaluate(self, x, out, *args, **kwargs):
        t1 = time.time()
        x_numeric = x.X if hasattr(x, "X") else x
        sim_id = abs(hash(tuple(x_numeric))) % (10**10) 
        
        # CHANGE self.mom_kit to self.m_kit
        w_mom = self.m_kit['scaler'].inverse_transform(
            self.m_kit['pca'].inverse_transform(x_numeric[0:4].reshape(1, -1))
        ).flatten()
        
        # CHANGE self.qual_kit to self.q_kit
        w_qual = self.q_kit['scaler'].inverse_transform(
            self.q_kit['pca'].inverse_transform(x_numeric[4:8].reshape(1, -1))
        ).flatten()
        w_mom, w_qual = np.clip(w_mom, 0, 1), np.clip(w_qual, 0, 1)

        opt_threshold = x_numeric[8]
        opt_beta = x_numeric[9]
        mom_decay = x_numeric[10]
        qual_decay = x_numeric[11]
        macro_weights = x_numeric[12:16]/sum(abs(x_numeric[12:16]))

        # SuppressOutput is removed here so you see the engine progress
        df_h_crash = self.base.run_blended_sim(w_mom, w_qual, opt_threshold, opt_beta, mom_decay, qual_decay, macro_weights, 'crash', sim_id)
        df_h_boom = self.base.run_blended_sim(w_mom, w_qual, opt_threshold, opt_beta, mom_decay, qual_decay, macro_weights, 'boom', sim_id)
        
        df_sim = pd.concat([df_h_crash, df_h_boom]).reset_index(drop=True)
        if df_sim.empty:
            out["F"] = [9999999] * 5
            return

        ticker_cols = [c for c in df_sim.columns if c not in ['sim_id', 'date', 'total_value', 'pct_change']]
        df_sim['total_value'] = df_sim[ticker_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        
        f1 = np.maximum(0, df_h_crash[ticker_cols].sum(axis=1).iloc[0] - df_h_crash[ticker_cols].sum(axis=1)).sum() if not df_h_crash.empty else 9999999
        df_2020 = df_sim[(df_sim['date'] >= '2020-02-01') & (df_sim['date'] <= '2020-06-01')]
        f2 = np.maximum(0, df_2020['total_value'].iloc[0] - df_2020['total_value']).sum() if not df_2020.empty else 9999999
        df_2022 = df_sim[(df_sim['date'] >= '2022-01-01') & (df_sim['date'] <= '2022-12-31')]
        f3 = np.maximum(0, df_2022['total_value'].iloc[0] - df_2022['total_value']).sum() if not df_2022.empty else 9999999
        f4 = df_sim.iloc[-1]['total_value']
        f5 = df_sim['total_value'].pct_change().dropna().quantile(ANNUAL_RISK_PERCENTILE)
        columns = [
            'sim_id',
            'f1_2008',
            'f2_2020',
            'f3_2022',
            'f4_terminal',
            'f5_annual_worst',
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
        values = [sim_id, f1, f2, f3, f4, f5] + list(x)
        df_out = pd.DataFrame({columns[i]: [values[i]] for i in range(len(columns))})

        save_result_agnostic(df_out, EVAL_FILE)


        out["F"] = [f1, f2, f3, -f4, -f5]
        print('time: {}'.format(time.time() - t1))

def main():
   
        
        
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

    problem = TripleThreatProblem(m_kit, q_kit, df_macro, data_features, df_price, params, training_periods, HOLDINGS_FILE)

    # Attempt to load existing progress
    algorithm = load_checkpoint(CHECKPOINT_URI)
    
    # Pack the re-injection data for the problem
    # Note: Added HOLDINGS_FILE here to match your current __init__
    problem_args = (m_kit, q_kit, df_macro, data_features, df_price, params, training_periods, HOLDINGS_FILE)

    if algorithm is None:
        print("Initial Startup: Building RANDOM population...")
        
        
        # 4. Initialize Manager and NSGA2 (Default sampling is FloatRandomSampling)
        problem = RobustParallelManager(NUM_WORKERS, TIMEOUT, TripleThreatProblem, problem_args)
        algorithm = NSGA2(
            pop_size=POP_SIZE,
            n_offsprings=N_OFFSPRINGS,
            eliminate_duplicates=True
        )
        algorithm.setup(problem, termination=('n_gen', GEN_COUNT), seed=1)
    else:
        print(f"Resuming from Generation {algorithm.n_gen}...")
        
        # 1. Initialize the fresh manager
        problem = RobustParallelManager(NUM_WORKERS, TIMEOUT, TripleThreatProblem, problem_args)
        
        # Sync the generation counts
        problem.n_gen = algorithm.n_gen 
        algorithm.problem = problem

        # 2. FORCE OVERRIDE TERMINATION
        # We re-import the termination factory to ensure it's fresh
        from pymoo.termination import get_termination
        algorithm.termination = get_termination("n_gen", GEN_COUNT)
        
        # 3. CRITICAL: Reset the 'has_finished' flag if it exists
        algorithm.has_terminated = False

    # --- 4. EXECUTION LOOP ---
    while algorithm.has_next():
        infills = algorithm.ask()
        print(f"--- Gen {algorithm.n_gen} Evaluation Start ---")
        
        # In a parallel version, you'd use your input_queue logic here. 
        # For this dry run, it uses the standard evaluator.
        algorithm.evaluator.eval(algorithm.problem, infills)
        
        algorithm.tell(infills=infills)
        
        # Checkpoint at the end of every successful generation
        save_checkpoint(algorithm, CHECKPOINT_URI)
        print(f"Gen {algorithm.n_gen} Success. Checkpoint saved.")

    print("--- Optimization Complete ---")
    algorithm.problem.cleanup()
    # At the very end of your script
    
    print("Optimization Complete")
    #os.system("sudo shutdown -h now")

if __name__ == "__main__":
     #limit_to_1_vcpu()
     
     #with monitor_peak_memory():
    main()