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
    ql_e_greedy = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=gamma, alpha=alpha, epsilon=0.1, epsilon_decay=0.99, n_iter=10000)
    ql_e_greedy.run()

    # Evaluate the performance (can be average reward, final reward, etc.)
    last_reward = ql_e_greedy.run_stats[-1]['Max V']
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

# Q-Learning with Epsilon-Greedy Exploration
ql_ε_greedy = hiive.mdptoolbox.mdp.QLearning(P, R, gamma = study.best_params["gamma"], alpha = study.best_params["alpha"],  epsilon=0.1, epsilon_decay=0.99, n_iter=10000)
ql_ε_greedy.run()

results = {
    "gamma": study.best_params["gamma"],
    "alpha": study.best_params["alpha"],
    "numIterations": 10000,
    "policy": ql_ε_greedy.policy,
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

file_name = "fr_e_greedy_optimal_results.pickle"

import pickle
with open(file_name, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(results)
print("Best parameters:", study.best_params)
