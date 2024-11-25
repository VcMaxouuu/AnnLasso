import os
import numpy as np
import pandas as pd
from harderLASSO.models import harderLASSORegressor
from harderLASSO.utils import StandardScaler, X_to_tensor, lambda_qut_regression
from torch.nn import ELU
from joblib import Parallel, delayed

def evaluate_model(model, dataset):
    X_train, y_train = dataset

    model.fit(X_train, y_train)
    selected_features = model.imp_feat[1]
    return selected_features

def simulate(X_train, errors, n, s, run_id):
    # Generate dataset
    features = pd.read_csv(f"simulations/parameters_analysis/data/features_{s}.csv")[f"{run_id}"]
    beta = 3 * np.ones(s)
    X_train_n = X_train.iloc[:n, :]
    errors_n = errors.iloc[:n, :]

    if features.empty:
        y_train = errors_n.iloc[:, run_id]
    else:
        y_train = np.dot(X_train_n.iloc[:, features], beta) + errors_n.iloc[:n, run_id]

    dataset = (X_train_n, y_train)

    scaled_X = StandardScaler().fit_transform(X_to_tensor(X_train))
    lambda_qut = lambda_qut_regression(scaled_X, ELU(alpha=1))

    # Evaluate each model and collect the results
    models = [
        (harderLASSORegressor(lambda_qut = 0.7*lambda_qut), 'harderLASSO ANN - 0.7lambda'),
        (harderLASSORegressor(lambda_qut = lambda_qut), 'harderLASSO ANN - lambda'),
        (harderLASSORegressor(lambda_qut = 1.3*lambda_qut), 'harderLASSO ANN - 1.3lambda'),
    ]

    result = {'run_id': run_id}
    for model, model_name in models:
        result[model_name] = evaluate_model(model, dataset)

    return result

if __name__ == "__main__":
    m = 100
    n_values = [20, 40, 60, 80, 100]
    s_values = range(21)
    X_train = pd.read_csv("simulations/parameters_analysis/data/X_train.csv")
    errors = pd.read_csv("simulations/parameters_analysis/data/errors.csv")


    for n in n_values:
        outdir = os.path.join(os.path.dirname(__file__), f'results/n{n}')
        os.makedirs(outdir, exist_ok=True)
        for s in s_values:
            print(f"Processing n = {n}, s = {s}")

            # Run m simulations in parallel for the current n
            results = Parallel(n_jobs=-1)(
                delayed(simulate)(X_train, errors, n, s, run_id) for run_id in range(m)
            )

            # Convert list of dicts to DataFrame
            results_df = pd.DataFrame(results)

            outname = f's{s}.csv'
            fullname = os.path.join(outdir, outname)
            results_df.to_csv(fullname, index=False, mode='w', header=True)

            print(f"Completed s = {n}. Results saved to {fullname}.")

    print("All simulations completed.")
