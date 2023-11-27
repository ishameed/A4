import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_name = '../hp_results/fr_hp_500_gamma.pickle'
with open(file_name, 'rb') as handle:
    results = pickle.load(handle)

for combo in range(500):
    run_num =3
    print("gamma: ", results["gamma_runs"][run_num][combo], "theta: ", results["theta_runs"][run_num][combo], "value: ", results["performance_metric_runs"][run_num][combo])

import optuna
from optuna.visualization import plot_optimization_history, plot_slice, plot_contour, plot_param_importances

study = results["study_instances"][0]

# Create or load an Optuna study
# study = optuna.create_study( ... )
# study.optimize( ... )

# Optimization history plot
fig = plot_optimization_history(study)
fig.show()

# Slice plot
fig = plot_slice(study)
fig.show()

# Contour plot of hyperparameters
fig = plot_contour(study)
fig.show()

# Hyperparameter importance plot
fig = plot_param_importances(study)
fig.show()
