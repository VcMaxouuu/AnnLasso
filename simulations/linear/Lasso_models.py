import os
import numpy as np
import pandas as pd
from harderLASSO.models import harderLASSORegressor
from harderLASSO.utils import StandardScaler, X_to_tensor, lambda_qut_regression
from torch.nn import ELU
from joblib import Parallel, delayed


def evaluate_model(model, dataset, lambda_qut):
    model.lambda_qut = lambda_qut

    X_train, y_train, X_test, y_test = dataset

    model.fit(X_train, y_train)
    selected_features = model.imp_feat[1]
    mu_hat = model.predict(X_test).squeeze()
    mse = (np.square(mu_hat - y_test.squeeze())).mean(axis=0)

    model_results = (selected_features, mse)
    return model_results

def simulate(X_train, X_test, errors, s, lambda_qut, run_id):
    # Generate dataset
    features = pd.read_csv(f"simulations/linear/data/features_{s}.csv")[f"{run_id}"]
    beta = 3 * np.ones(s)
    if features.empty:
        y_train = errors.iloc[:, run_id]
        y_test = np.zeros(shape=(X_test.shape[0]))
    else:
        y_train = np.dot(X_train.iloc[:, features], beta) + errors.iloc[:, run_id]
        y_test = np.dot(X_test.iloc[:, features], beta)

    dataset = (X_train, y_train, X_test, y_test)

    # Evaluate each model and collect the results
    models = [
        (harderLASSORegressor(hidden_dims=(20,), nu=0.1), 'harderLASSO ANN'),
        (harderLASSORegressor(hidden_dims=(20,), nu=None), 'LASSO ANN'),
        (harderLASSORegressor(hidden_dims=None, nu=0.1), 'harderLASSO linear'),
        (harderLASSORegressor(hidden_dims=None, nu=None), 'LASSO linear')
    ]

    result = {'run_id': run_id}
    for model, model_name in models:
        model_results = evaluate_model(model, dataset, lambda_qut)
        result[model_name] = model_results

    return result

if __name__ == "__main__":
    m = 100
    s_values = range(31)
    X_train = pd.read_csv("simulations/linear/data/X_train.csv")
    X_test = pd.read_csv("simulations/linear/data/X_test.csv")
    errors = pd.read_csv("simulations/linear/data/errors.csv")

    # Compute lambda_qut once since it's the same for linear and one layer ANN
    scaled_X = StandardScaler().fit_transform(X_to_tensor(X_train))
    lambda_qut = lambda_qut_regression(scaled_X, ELU(alpha=1))


    outdir = os.path.join(os.path.dirname(__file__), 'results/Lasso_models')
    os.makedirs(outdir, exist_ok=True)

    for s in s_values:
        print(f"Processing s = {s}...")

        # Run m simulations in parallel for the current s
        results = Parallel(n_jobs=-1)(
            delayed(simulate)(X_train, X_test, errors, s, lambda_qut, run_id) for run_id in range(m)
        )

        results_df = pd.DataFrame(results)
        outname = f's{s}.csv'
        fullname = os.path.join(outdir, outname)
        results_df.to_csv(fullname, index=False, mode='w', header=True)

        print(f"Completed s = {s}. Results saved to {fullname}.")

    print("All simulations completed.")
