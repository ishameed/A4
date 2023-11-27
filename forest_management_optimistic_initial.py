import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp

import optuna
from optuna.visualization import plot_slice, plot_contour, plot_param_importances


# Define the optimization function
def optimize_q_learning(trial):
    # Suggested values for gamma and alpha
    alpha = trial.suggest_float("alpha", 0.1, 0.5)
    gamma = trial.suggest_discrete_uniform('gamma', 0.5, 0.9999, 0.001)

    # Generate the Forest Management problem
    P, R = hiive.mdptoolbox.example.forest()

    # Run Q-Learning

    # Initialize the QLearning algorithm
    qlearner = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=gamma, alpha=alpha, n_iter=10000)

    # Set optimistic initial values, for example, 1.0, which assumes that all states are good before learning anything.
    # The actual optimistic value should be higher than the maximum expected reward.
    qlearner.Q = np.ones(qlearner.Q.shape) * 1.0
    qlearner.run()

    # Evaluate the performance (can be average reward, final reward, etc.)
    last_reward = qlearner.run_stats[-1]['Max V']
    return last_reward

# Generate the Forest Management problem
P, R = hiive.mdptoolbox.example.forest(S=10, r1=100, r2=10, p=0.1)

# Create an Optuna study and optimize
study = optuna.create_study(direction="maximize")
study.optimize(optimize_q_learning, n_trials=500)


# Slice plot
fig = plot_slice(study)
fig.show()

# Contour plot of hyperparameters
fig = plot_contour(study)
fig.show()

# Best parameters found
# print("Best parameters:", study.best_params)

qlearner = hiive.mdptoolbox.mdp.QLearning(P, R, gamma = study.best_params["gamma"], alpha = study.best_params["alpha"], n_iter=10000)

# Set optimistic initial values, for example, 1.0, which assumes that all states are good before learning anything.
# The actual optimistic value should be higher than the maximum expected reward.
qlearner.Q = np.ones(qlearner.Q.shape) * 1.0
qlearner.run()

results = {
    "gamma": study.best_params["gamma"],
    "alpha": study.best_params["alpha"],
    "numIterations": 10000,
    "policy": qlearner.policy,
}

results["maxV"] = []
results["avgV"] = []
results["reward"] = []
results["wallClockTime"] = []

for iteration in ql_ε_greedy.run_stats:
    results["maxV"].append(iteration["Max V"])
    results["avgV"].append(iteration["Mean V"])
    results["reward"].append(iteration["Reward"])
    results["wallClockTime"].append(iteration["Time"])

file_name = "fr__optimal_optimistic_initial_results.pickle"

import pickle
with open(file_name, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(results)
print("Best parameters:", study.best_params)
