import numpy as np
from sklearn.preprocessing import StandardScaler

def are_standard_scalers_equivalent(scaler1, scaler2, rtol=1e-5, atol=1e-8) -> bool:
    """
    Determines if two sklearn StandardScaler objects are equivalent by checking
    their hyperparameters and any learned parameters (if fitted).
    """
    # 1. Check if they are actually both StandardScaler objects
    if not (isinstance(scaler1, StandardScaler) and isinstance(scaler2, StandardScaler)):
        return False
        
    # 2. Check user-defined initialization hyperparameters
    hyperparams = ['with_mean', 'with_std', 'copy']
    for param in hyperparams:
        if getattr(scaler1, param) != getattr(scaler2, param):
            return False

    # 3. Identify which learned attributes are present
    # Learned attributes in sklearn always end with a trailing underscore
    fitted_attrs = ['mean_', 'var_', 'scale_', 'n_samples_seen_']
    
    has_attrs1 = [hasattr(scaler1, attr) for attr in fitted_attrs]
    has_attrs2 = [hasattr(scaler2, attr) for attr in fitted_attrs]
    
    # If one is fitted and the other isn't, they aren't equivalent
    if has_attrs1 != has_attrs2:
        return False
        
    # If neither is fitted, they are equivalent based purely on hyperparameters
    if not any(has_attrs1):
        return True

    # 4. Compare the learned parameters numerically
    # n_samples_seen_ can be an int or an array depending on features, check it first
    if not np.allclose(scaler1.n_samples_seen_, scaler2.n_samples_seen_, rtol=rtol, atol=atol):
        return False

    # Check mean_ and var_ if they exist (depends on with_mean and with_std settings)
    if scaler1.with_mean:
        if not np.allclose(scaler1.mean_, scaler2.mean_, rtol=rtol, atol=atol):
            return False
            
    if scaler1.with_std:
        if not np.allclose(scaler1.var_, scaler2.var_, rtol=rtol, atol=atol) or \
           not np.allclose(scaler1.scale_, scaler2.scale_, rtol=rtol, atol=atol):
            return False

    return True


import numpy as np
from sklearn.decomposition import PCA

def are_pca_models_equivalent(pca1, pca2, rtol=1e-5, atol=1e-8) -> bool:
    """
    Determines if two sklearn PCA objects are functionally equivalent,
    accounting for hyperparameters, learned attributes, and sign indeterminacy
    of components.
    """
    # 1. Check if they are both PCA objects
    if not (isinstance(pca1, PCA) and isinstance(pca2, PCA)):
        return False
        
    # 2. Check key initialization hyperparameters
    hyperparams = ['n_components', 'whiten', 'svd_solver', 'tol', 'iterated_power', 'random_state']
    for param in hyperparams:
        if getattr(pca1, param) != getattr(pca2, param):
            return False

    # 3. Check fit status
    # 'components_' is only created after a successful fit
    has_fit1 = hasattr(pca1, 'components_')
    has_fit2 = hasattr(pca2, 'components_')
    
    if has_fit1 != has_fit2:
        return False
    if not has_fit1:  # Both are unfitted, but hyperparameters match
        return True

    # 4. Compare basic learned parameters (Removed 'n_samples_seen_')
    basic_attrs = ['mean_', 'explained_variance_', 'explained_variance_ratio_']
    for attr in basic_attrs:
        val1 = getattr(pca1, attr)
        val2 = getattr(pca2, attr)
        # Handle cases where attributes might be None
        if (val1 is None) != (val2 is None):
            return False
        if val1 is not None:
            if not np.allclose(val1, val2, rtol=rtol, atol=atol):
                return False

    # 5. Compare components_ (eigenvectors) handling sign flipping
    if pca1.components_.shape != pca2.components_.shape:
        return False

    for row1, row2 in zip(pca1.components_, pca2.components_):
        direct_match = np.allclose(row1, row2, rtol=rtol, atol=atol)
        inverted_match = np.allclose(row1, -row2, rtol=rtol, atol=atol)
        if not (direct_match or inverted_match):
            return False

    # 6. Check noise variance if it exists
    if hasattr(pca1, 'noise_variance_'):
        if not np.allclose(pca1.noise_variance_, pca2.noise_variance_, rtol=rtol, atol=atol):
            return False

    return True


import pandas as pd

def are_dataframes_equivalent(df1, df2, check_dtype=True, check_index_type=True, rtol=1e-5, atol=1e-8) -> bool:
    """
    Determines if two pandas DataFrames are equivalent.
    
    Parameters:
    - check_dtype: If True, columns must have the exact same data type (e.g., float32 vs float64 will fail).
    - check_index_type: If True, the Index/MultiIndex types must match exactly.
    - rtol, atol: Floating-point tolerances for numerical columns.
    """
    # 1. Quick instance check
    if not (isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame)):
        return False
        
    # 2. Use pandas native testing utility to catch mismatches
    try:
        pd.testing.assert_frame_equal(
            df1, 
            df2, 
            check_dtype=check_dtype, 
            check_index_type=check_index_type,
            rtol=rtol,
            atol=atol
        )
        return True
    except AssertionError:
        return False
    

import xarray as xr

def are_dataarrays_equivalent(da1, da2, method='equals', rtol=1e-5, atol=1e-8) -> bool:
    """
    Determines if two xarray DataArrays are equivalent based on the chosen strictness.
    
    Parameters:
    - method: 'equals', 'identical', 'broadcast_equals', or 'allclose'
        * 'equals': Matches dimensions, coordinates, and values (handles NaNs safely). 
                    Ignores DataArray names and metadata attributes.
        * 'identical': Matches everything 'equals' does, PLUS requires the array name 
                       and all metadata attributes (`.attrs`) to match exactly.
        * 'broadcast_equals': Matches if the arrays are equal *after* broadcasting 
                              them to share the same dimensions.
        * 'allclose': Matches dimensions and coordinates, but checks data values using 
                      floating-point tolerances (rtol/atol). Ignores attributes and names.
    - rtol, atol: Floating-point tolerances used ONLY if method='allclose'.
    """
    # 1. Instance validation
    if not (isinstance(da1, xr.DataArray) and isinstance(da2, xr.DataArray)):
        return False
        
    # 2. Evaluate using built-in exact match methods
    if method == 'equals':
        return da1.equals(da2)
        
    elif method == 'identical':
        return da1.identical(da2)
        
    elif method == 'broadcast_equals':
        return da1.broadcast_equals(da2)
        
    # 3. Evaluate using floating-point tolerances (Requires testing utility)
    elif method == 'allclose':
        try:
            xr.testing.assert_allclose(da1, da2, rtol=rtol, atol=atol)
            return True
        except AssertionError:
            return False
            
    else:
        raise ValueError("Method must be one of: 'equals', 'identical', 'broadcast_equals', 'allclose'")
    

import numpy as np

def are_arrays_equivalent(arr1, arr2, rtol=1e-5, atol=1e-8) -> bool:
    """
    Determines if two NumPy arrays are equivalent. Handles float precision 
    and treats NaN values as equivalent if they are in the same position.
    """
    # 1. Quick instance check
    if not (isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray)):
        return False

    # 2. Shape and type must match
    if arr1.shape != arr2.shape:
        return False

    # 3. Handle Floating-Point arrays (allow for tiny precision drifts)
    if np.issubdtype(arr1.dtype, np.floating) or np.issubdtype(arr2.dtype, np.floating):
        # equal_nan=True ensures that np.nan == np.nan evaluates to True
        return np.allclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True)

    # 4. Handle Integer, Boolean, String, or Object arrays (exact match)
    else:
        # For object arrays containing NaN, standard array_equal can fail.
        # equal_nan=True handles it safely across modern NumPy versions.
        try:
            return np.array_equal(arr1, arr2, equal_nan=True)
        except TypeError:
            # Fallback for complex object arrays where equal_nan isn't supported
            return np.array_equal(arr1, arr2)
        

import numpy as np
import pandas as pd

def are_lists_equivalent(list1, list2, ignore_order=False) -> bool:
    """
    Determines if two lists are equivalent.
    
    Parameters:
    - ignore_order: If True, compares lists as sets/multisets (order won't matter).
    """
    # 1. Basic Type and Length checks
    if not (isinstance(list1, list) and isinstance(list2, list)):
        return False
    if len(list1) != len(list2):
        return False

    # 2. Handle Order-Insensitive Comparison
    if ignore_order:
        # We can't just use set() because sets destroy duplicate elements,
        # and we can't use sorted() because lists might contain unorderable types (like dicts).
        # Instead, we copy list2 and match/remove elements one by one.
        pool = list(list2)
        for item1 in list1:
            match_index = None
            for idx, item2 in enumerate(pool):
                if _are_items_equivalent(item1, item2):
                    match_index = idx
                    break
            if match_index is not None:
                pool.pop(match_index)
            else:
                return False
        return len(pool) == 0

    # 3. Handle Order-Sensitive Comparison (Default)
    else:
        for item1, item2 in zip(list1, list2):
            if not _are_items_equivalent(item1, item2):
                return False
        return True


def _are_items_equivalent(item1, item2) -> bool:
    """Helper to safely compare items, routing complex types to their correct checks."""
    # Check types match first
    if type(item1) is not type(item2):
        return False
        
    # Route NumPy arrays
    if isinstance(item1, np.ndarray):
        return np.array_equal(item1, item2, equal_nan=True)
        
    # Route Pandas DataFrames
    if isinstance(item1, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(item1, item2)
            return True
        except AssertionError:
            return False
            
    # Standard Python equality (handles strings, ints, nested dicts/lists without custom objects)
    try:
        # Check for NaN safety in standard python floats
        if isinstance(item1, float) and np.isnan(item1) and np.isnan(item2):
            return True
        return item1 == item2
    except (ValueError, TypeError):
        return False
    

from functools import partial
import numpy as np
import pandas as pd

def are_partials_equivalent(p1, p2) -> bool:
    """
    Determines if two functools.partial objects are equivalent by checking
    if they wrap the same function and hold identical arguments.
    """
    # 1. Ensure both are actually partial objects
    if not (isinstance(p1, partial) and isinstance(p2, partial)):
        return False

    # 2. Check if they wrap the exact same underlying function
    if p1.func != p2.func:
        return False

    # 3. Check positional arguments (args)
    if len(p1.args) != len(p2.args):
        return False
    for arg1, arg2 in zip(p1.args, p2.args):
        if not _safe_element_compare(arg1, arg2):
            return False

    # 4. Check keyword arguments (keywords)
    # Normalize None to an empty dict for clean comparison
    kw1 = p1.keywords if p1.keywords is not None else {}
    kw2 = p2.keywords if p2.keywords is not None else {}

    if kw1.keys() != kw2.keys():
        return False

    for key, val1 in kw1.items():
        val2 = kw2[key]
        if not _safe_element_compare(val1, val2):
            return False

    return True


def _safe_element_compare(item1, item2) -> bool:
    """Helper to deeply compare argument values without crashing on complex types."""
    if type(item1) is not type(item2):
        return False

    # Handle NumPy Arrays
    if isinstance(item1, np.ndarray):
        return np.array_equal(item1, item2, equal_nan=True)

    # Handle DataFrames
    if isinstance(item1, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(item1, item2)
            return True
        except AssertionError:
            return False

    # Handle nested collections (lists/dicts)
    if isinstance(item1, (list, dict)):
        try:
            # Quick fallback to standard equality if they are simple collections
            return item1 == item2
        except ValueError:
            # If standard equality fails (e.g., contains an array inside a list)
            if isinstance(item1, list):
                if len(item1) != len(item2): return False
                return all(_safe_element_compare(i1, i2) for i1, i2 in zip(item1, item2))
            return False

    # Standard Python equality (with NaN fallback for standard floats)
    try:
        if isinstance(item1, float) and np.isnan(item1) and np.isnan(item2):
            return True
        return item1 == item2
    except (ValueError, TypeError):
        return False