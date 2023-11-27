import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp
import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

np.random.seed(42)
# optuna.seed_rng(42)

file_name = '../hp_results/fl_hp_30_gamma.pickle'
# file_name = 'no'

grid_size = 30
hole_states = []

def create_frozen_lake_grid(grid_size, hole_probability):
    """
    Create a Frozen Lake environment with a specified grid size and hole probability.
    Holes are randomly placed in the grid, except for the start and goal positions.
    The start state is top-left (0,0), and the goal state is bottom-right (grid_size-1, grid_size-1).

    :param grid_size: Size of the grid (e.g., 4 for a 4x4 grid).
    :param hole_probability: Probability of a cell being a hole.
    :return: P (transition probabilities), R (rewards).
    """
    n_states = grid_size * grid_size
    n_actions = 4  # Up, Down, Left, Right

    # Initialize P and R
    P = np.zeros((n_actions, n_states, n_states))
    R = np.zeros((n_states, n_actions))

    # Define the goal state
    goal_state = n_states - 1

    # Randomly place holes in the grid, avoiding the start (0) and goal positions
    hole_states = set()
    for state in range(1, n_states - 1):
        if np.random.rand() < hole_probability:
            hole_states.add(state)

    for action in range(n_actions):
        for state in range(n_states):
            if state == goal_state or state in hole_states:
                # Terminal state: remain in the same state
                P[action][state][state] = 1
                continue

            # Compute new state based on action
            x, y = divmod(state, grid_size)
            if action == 0 and x > 0: y -= grid_size  # Up
            elif action == 1 and x < grid_size - 1: y += grid_size  # Down
            elif action == 2 and y > 0: y -= 1  # Left
            elif action == 3 and y < grid_size - 1: y += 1  # Right

            new_state = x * grid_size + y

            # Assign transition probability and rewards
            P[action][state][new_state] = 1
            R[state][action] = -0.01  # Small penalty for each move to encourage efficiency
            if new_state in hole_states:
                R[state][action] = -1  # Penalty for falling into a hole
            elif new_state == goal_state:
                R[state][action] = 1  # Reward for reaching the goal

    return P, R, hole_states


def run_frozen_lake(P, R, gamma, theta):
    # Create and run the MDP model using Value Iteration
    
    vi = hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma, epsilon=theta)
    vi.run()
    # You could return other metrics here, such as mean reward, etc.
    # print("AVERAGE VALUE: ", np.mean(vi.V))
    # print(vi.V)
    return np.max(vi.V)

def objective(trial):
    # Define the hyperparameters to tune
    theta = trial.suggest_discrete_uniform('theta', 1e-5, 1e-3, 1e-5)
    gamma = trial.suggest_discrete_uniform('gamma', 0.5, 0.9999, 0.001)

    P, R, hole_states = create_frozen_lake_grid(grid_size, 0.10)
    hole_states = hole_states
    
    
    # Run the MDP model with the given hyperparameters
    avg_value = run_frozen_lake(P, R, gamma, theta)
    
    # Negative average value as we want to maximize it, but Optuna minimizes the objective
    return -avg_value

# Create a study object and specify the direction is 'minimize'
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)

num_optimization_runs = 1
gamma_runs = {}
theta_runs = {}
performance_metric_runs = {}
study_best_params = {}
study_instances = {}
hole_states = {}

plt.figure(figsize=(10, 6))
plt.xlabel('Gamma')
plt.ylabel('Performance Metric')
plt.title('Performance Metric vs. Gamma')
plt.grid(True)
for run_num in range(num_optimization_runs):
    # sampler = optuna.samplers.GridSampler(search_space)
    # study = optuna.create_study(direction='maximize', sampler=sampler)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=run_num), pruner = pruner)
    # study.optimize(objective, n_trials=len(search_space['gamma']) * len(search_space['theta']))
    study.optimize(objective, n_trials=500)

    study_best_params[run_num] = study.best_params
    study_instances[run_num] = study
    gammas = [trial.params['gamma'] for trial in study.trials]
    gamma_runs[run_num] = gammas

    thetas = [trial.params['theta'] for trial in study.trials]
    theta_runs[run_num] = thetas

    performance_metrics = [-trial.value for trial in study.trials]
    performance_metric_runs[run_num] = performance_metrics

    study_best_params[run_num]["performance_metric"] = max(performance_metrics)
    hole_states[run_num] = hole_states


print("BEST PARAMS: ", study_best_params)

results = {
    "gamma_runs": gamma_runs,
    "theta_runs": theta_runs,
    "performance_metric_runs": performance_metric_runs,
    "study_best_params": study_best_params,
    "study_instances": study_instances,
    "hole_states": hole_states
}

import pickle
with open(file_name, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# plt.show()

# Best hyperparameters:  {'gamma': 0.9998910004702775, 'theta': 0.0001964584973819517}