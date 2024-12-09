"""
Provides utilities for different task types.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_baseline_survival_function(
    predictions: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray
) -> pd.Series:
    r"""
    Computes the baseline survival function :math:`(S_0(t))` for a Cox proportional hazards model.

    :param predictions: Predicted log hazard ratios for each individual.
    :type predictions: np.ndarray
    :param durations: Observed times until event or censoring for each individual.
    :type durations: np.ndarray
    :param events: Event indicators (1 if event occurred, 0 if censored) for each individual.
    :type events: np.ndarray
    :return: Baseline survival function indexed by time.
    :rtype: pd.Series
    """
    risk_scores = np.exp(predictions)

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'duration': durations,
        'event': events,
        'risk': risk_scores
    })

    # Group data by unique durations
    df = df.groupby('duration').agg({
        'risk': 'sum',
        'event': 'sum'
    })

    # Replace risk with its cumulative sum from highest to lowest duration.
    df = df.sort_index(ascending=False)
    df = df.assign(risk=lambda x: x['risk'].cumsum())

    # Compute baseline hazard at each duration
    df['baseline_hazard'] = df['event'] / df['risk']

     # Compute comultative baseline hazard at each duration
    df = df.sort_index(ascending=True)
    df['cumulative_baseline_hazard'] = df['baseline_hazard'].cumsum()

    # Calculate the baseline survival function
    df['baseline_survival'] = np.exp(-df['cumulative_baseline_hazard'])

    return df["baseline_survival"]

def compute_individual_survival_functions(
    baseline_survival: pd.Series,
    predictions: np.ndarray
) -> pd.DataFrame:
    """
    Computes the individual survival functions for all individuals in `predictions`.

    :param baseline_survival: Baseline survival function computed from `compute_baseline_survival_function`.
    :type baseline_survival: pd.Series
    :param predictions: Predicted log hazard ratios for each individual.
    :type predictions: np.ndarray
    :return: DataFrame where each row corresponds to an individual and each column to a duration,
             containing the survival probabilities for each individual at each duration.
    :rtype: pd.DataFrame
    """
    S0_t = baseline_survival.values
    risk_scores = np.exp(predictions)
    survival_matrix = np.array([S0_t**risk for risk in risk_scores])
    survival_df = pd.DataFrame(survival_matrix, columns=baseline_survival.index)

    return survival_df


def plot_baseline_survival(baseline_survival):
    """
    Plots the baseline survival function.

    :param baseline_survival: Baseline survival function computed from `compute_baseline_survival_function`.
    :type baseline_survival: pd.Series
    """
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(baseline_survival.index, baseline_survival.values, label="Baseline Survival", color='black', linestyle='--')
    plt.title("Baseline Survival Function")
    plt.ylabel(r"$S_0(t)$")
    plt.xlabel("Time")
    plt.legend()
    plt.show()


def plot_survival_functions(survival_functions):
    """
    Plots the survival functions for all individuals in `survival_functions`.

    :param survival_functions: DataFrame where each row corresponds to an individual and each column to a duration,
                                containing the survival probabilities for each individual at each duration.
    :type survival_functions: pd.DataFrame
    """
    plt.figure(figsize=(10, 6), dpi=200)
    for i, survival_curve in survival_functions.iterrows():
        plt.plot(survival_functions.columns, survival_curve, label=f'Individual {i}', alpha=0.8)

    plt.title("Individual Survival Functions")
    plt.xlabel("Time")
    plt.ylabel(r"$S(t)$")
    plt.legend(loc="upper right")
    plt.show()
