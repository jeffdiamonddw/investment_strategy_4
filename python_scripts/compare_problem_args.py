import joblib
from comparisons import *

with open('temp/problem_args1.joblib', 'rb') as fp1:
    problem_args_1 = joblib.load(fp1)
with open('temp/problem_args2.joblib', 'rb') as fp2:
    problem_args_2 = joblib.load(fp2)


compare_functions = {
    "<class 'sklearn.preprocessing._data.StandardScaler'>": are_standard_scalers_equivalent,
    "<class 'sklearn.decomposition._pca.PCA'>": are_pca_models_equivalent,
    "<class 'pandas.DataFrame'>": are_dataframes_equivalent,
    "<class 'xarray.core.dataarray.DataArray'>": are_dataarrays_equivalent,
    "<class 'numpy.ndarray'>": are_arrays_equivalent,
    "<class 'functools.partial'>": are_partials_equivalent
}

for i in range(len(problem_args_1)):
    if isinstance(problem_args_1[i], dict):
        for key in problem_args_1[i]:
            value = problem_args_1[i][key]
            if str(value.__class__) in compare_functions:
                func = compare_functions[str(value.__class__)]
                result = func(problem_args_1[i][key], problem_args_2[i][key])
                print(i, key, value.__class__, result) 
            else:
                print(i, key, value.__class__, problem_args_1[i][key] == problem_args_2[i][key])
    elif isinstance(problem_args_1[i], list):
        print(i, 'list', problem_args_1[i][0].__class__, are_lists_equivalent(problem_args_1[i], problem_args_2[i]))
    else:
        value = problem_args_1[i]
        if str(value.__class__) in compare_functions:
                func = compare_functions[str(value.__class__)]
                result = func(problem_args_1[i], problem_args_2[i])
                print(i, value.__class__, result) 
        else:
                print(i, value.__class__, problem_args_1[i] == problem_args_2[i])



