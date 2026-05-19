import functools
import time, random
import os

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling



from triple_threat import RobustParallelManager, load_checkpoint, save_checkpoint
from pareto_navigator import ParetoNavigator, mean_annual_drawdown_integral, annualized_return, pct_change_quantile, worst_annual_return
from triple_threat import get_triple_threat_params, TripleThreatProblem

  



TICKER_FILE = "strategy/multi_dim_stock_list.csv"
HOLDINGS_FOLDER = "s3://jdinvestment/median_17_year_holdings"
EVAL_FOLDER = "s3://jdinvestment/median_17_year_evaluations"
INITIAL_POP_FILE = "analysis/top_ranked.csv"
CHECKPOINT_URI = "s3://jdinvestment/checkpoints/median_17_year_checkpoint"
POP_SIZE = 180
N_OFFSPRING = 188
NUM_WORKERS = 94
TARGET_COMPLETIONS = 90
GEN_COUNT = 150
TIMEOUT_SEC = 180 * 3
OUT_FOLDER = "s3://jdinvestment/median_17_year_output"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

class RobustMedianEvaluator(ElementwiseProblem):
    def __init__(self, workhorse_cls, workhorse_args, n_samples=5, perturbation_cv=0.01, out_folder = OUT_FOLDER):
        """
        An 'Abstract-Aware' wrapper. It stores the CLASS of the simulation 
        to be run, not the object.
        """
        # 1. Instantiate the simulation locally to determine dimensions
        self.local_sim = workhorse_cls(*workhorse_args)
        self.n_samples = n_samples
        self.perturbation_cv = perturbation_cv
        self.workhorse_args = workhorse_args
        self.out_folder = out_folder
        
        # 2. Inherit dimensions from the actual simulation
        super().__init__(
            n_var=self.local_sim.n_var,
            n_obj=self.local_sim.n_obj,
            n_constr=self.local_sim.n_constr,
            xl=self.local_sim.xl,
            xu=self.local_sim.xu
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Implementation logic for running 5 simulations and averaging[cite: 1]
        sim_id = abs(hash(tuple(x))) % (10**10)
        results = []
        for _ in range(self.n_samples):
            # Apply Gaussian perturbation based on input CV[cite: 1]
            noise = np.random.normal(0, self.perturbation_cv * np.abs(x), size=x.shape)
            perturbed_x = np.clip(x + noise, self.xl, self.xu)
            
            sample_out = {}
            # Route evaluation to the locally held simulation object
            self.local_sim._evaluate(perturbed_x, sample_out, *args, **kwargs)
            results.append(sample_out["F"])

        median_result = np.median(results, axis=0)
        param_cols = ["p_{}".format(i) for i in range(len(x))]
        objective_cols = list(self.workhorse_args[-3].keys())
        df_out = pd.DataFrame([[sim_id] + list(-median_result) + list(x)], columns = ['sim_id'] + objective_cols + param_cols)
        time.sleep(random.uniform(0, 5))
        df_out.to_csv("{}/{}.csv".format(self.out_folder, sim_id))    
        out["F"] = median_result



if __name__ == "__main__":

    df_starting_pop = pd.read_csv(INITIAL_POP_FILE)
    df_pop = df_starting_pop.loc[df_starting_pop['rank'] == 0]


    weight_cols = [
        'dollar_ret_1p', 'dollar_ret_6p', 'dollar_ret_13p',
       'dollar_ret_26p', 'avg_eps_1q', 'avg_eps_2q', 'avg_eps_4q',
       'avg_eps_8q', 'threshold', 'beta', 'mom_decay', 'qual_decay',
       'macro_weights_0', 'macro_weights_1', 'macro_weights_2',
       'macro_weights_3'
    ]

    X_initial = df_pop[weight_cols].values
    initial_pop = Population.new("X", X_initial)
    
    periods = {
        '17_year': {'train_start_date': pd.to_datetime('Jan 1, 2005'), 'val_start_date': pd.to_datetime('Jan 1, 2008'), 'end_date': pd.to_datetime('Jan 1, 2025')},
    }
    
    
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
    
    
    problem = RobustParallelManager(
            num_workers=NUM_WORKERS,
            timeout_sec=TIMEOUT_SEC,
            workhorse_cls=RobustMedianEvaluator, # The wrapper class
            workhorse_args= (TripleThreatProblem, problem_args, 3, 0.01),      # The 'Wrapped' class + Sim Args
            var_count=16,
            obj_count=3,
            xl = xl,
            xu = xu,
            target_completions = TARGET_COMPLETIONS
    )
    

    # Attempt to load existing progress
    algorithm = load_checkpoint(CHECKPOINT_URI)
    
    if algorithm is None:
        print("Initial Startup: Building RANDOM population...")
        
        
        # 4. Initialize Manager and NSGA2 (Default sampling is FloatRandomSampling)
        # 1. Initialize the fresh manager
        

        algorithm = NSGA2(
            pop_size=POP_SIZE,
            n_offsprings=N_OFFSPRING,
            sampling = initial_pop,
            eliminate_duplicates=True
        )
        algorithm.setup(problem, termination=('n_gen', GEN_COUNT), seed=1)
    else:
        print(f"Resuming from Generation {algorithm.n_gen}...")
        
        
        
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
    
    
    
    