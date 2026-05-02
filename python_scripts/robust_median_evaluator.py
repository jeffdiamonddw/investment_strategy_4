import time, random
import os

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population

from manifold_dry_run_parallel import RobustParallelManager, load_checkpoint, save_checkpoint
from pareto_navigator import ParetoNavigator, get_pareto_nav_params

  




HOLDINGS_FOLDER = "s3://jdinvestment/robust_median_nav_holdings_4"
EVAL_FOLDER = "s3://jdinvestment/robust_median_nav_evaluations_4"
STAR_FILE = 'analysis/stars.csv'
INITIAL_POP_FILE = "temp/robust_mean_starting_pop.csv"
CHECKPOINT_URI = "s3://jdinvestment/checkpoints/robust_median_nav_checkpoint_4"
POP_SIZE = 180
N_OFFSPRING = 188
NUM_WORKERS = 94
TARGET_COMPLETIONS = 90
GEN_COUNT = 150
TIMEOUT_SEC = 180 * 5
OUT_FOLDER = "s3://jdinvestment/robust_nav_medians_4"

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
    param_cols = ["p_{}".format(i) for i in range(17)]
    X_initial = df_starting_pop[param_cols].values
    
    pareto_nav_args = get_pareto_nav_params(holdings_folder = HOLDINGS_FOLDER,  eval_folder = EVAL_FOLDER, star_file = STAR_FILE)
    xl, xu = pareto_nav_args[-2:]
    
    
    problem = RobustParallelManager(
            num_workers=NUM_WORKERS,
            timeout_sec=TIMEOUT_SEC,
            workhorse_cls=RobustMedianEvaluator, # The wrapper class
            workhorse_args= (ParetoNavigator, pareto_nav_args, 5, 0.01),      # The 'Wrapped' class + Sim Args
            var_count=17,
            obj_count=2,
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
            sampling = X_initial[:POP_SIZE],
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
    
    
    
    