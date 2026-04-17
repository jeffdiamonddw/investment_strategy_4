import os
import boto3
import pickle
import fsspec
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem

# --- THE FIX: CHOICE B ---
# This assumes manifold_dry_run_parallel.py is in the same directory.
# We import these so pickle knows how to 'hydrate' the old algorithm object.
try:
    from manifold_dry_run_parallel import RobustParallelManager, TripleThreatProblem
except ImportError:
    print("Error: Could not find manifold_dry_run_parallel.py in the current directory.")
    print("Fallback: Defining dummy classes to allow unpickling...")
    class RobustParallelManager: pass
    class TripleThreatProblem: pass

# --- CONFIGURATION ---
S3_INPUT_URI = "s3://jdinvestment/checkpoints/checkpoint_3.pkl"
S3_OUTPUT_URI = "s3://jdinvestment/checkpoints/checkpoint_4.pkl"
NEW_POP_SIZE = 90
NEW_OFFSPRING_SIZE = 94

class DummyProblem(Problem):
    """
    Temporary problem class to initialize the new algorithm dimensions.
    """
    def __init__(self):
        super().__init__(n_var=10, n_obj=5, n_constr=0, xl=0, xu=1)

def pivot_checkpoint():
    print(f"Reading old checkpoint from {S3_INPUT_URI}...")
    
    # 1. Load the old algorithm object
    with fsspec.open(S3_INPUT_URI, "rb") as f:
        old_algorithm = pickle.load(f)
    
    print(f"Successfully loaded Gen {old_algorithm.n_gen} from checkpoint.")
    
    # 2. Extract and Sort Population
    # We take the first 90. In NSGA-2, 'pop' is typically maintained 
    # such that the best-ranked individuals are at the front.
    old_pop = old_algorithm.pop
    print(f"Original population size: {len(old_pop)}")
    
    # Slice the top 90
    new_pop_data = old_pop[:NEW_POP_SIZE]
    print(f"Selected top {len(new_pop_data)} individuals for pivot.")
    
    # 3. Initialize the New Algorithm Object
    # We use DummyProblem just to satisfy the .setup() requirements.
    # The real RobustParallelManager will be injected when you run your main script.
    dummy_problem = DummyProblem()
    
    new_algorithm = NSGA2(
        pop_size=NEW_POP_SIZE,
        n_offsprings=NEW_OFFSPRING_SIZE,
        sampling=new_pop_data,
        eliminate_duplicates=True
    )
    
    # Setup populates the internal matrices for the new 90/94 dimensions
    new_algorithm.setup(
        dummy_problem, 
        termination=('n_gen', 300), 
        seed=1
    )
    
    # 4. Synchronize Meta-data
    # Ensure the generation count carries over
    new_algorithm.n_gen = old_algorithm.n_gen
    
    # 5. Save to S3
    print(f"Saving pivoted checkpoint to {S3_OUTPUT_URI}...")
    with fsspec.open(S3_OUTPUT_URI, "wb") as f:
        pickle.dump(new_algorithm, f)
    
    print("\n--- PIVOT SUCCESSFUL ---")
    print(f"New Population Size: {new_algorithm.pop_size}")
    print(f"New Offspring Size: {new_algorithm.n_offsprings}")
    print(f"Resuming from Gen: {new_algorithm.n_gen}")

if __name__ == "__main__":
    pivot_checkpoint()