import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp

import optuna
from optuna.visualization import plot_slice, plot_contour, plot_param_importances
import numpy as np

grid_size = 25

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

# Define the optimization function
def optimize_q_learning(trial):
    # Suggested values for gamma and alpha
    alpha = trial.suggest_float("alpha", 0.1, 0.5)
    gamma = trial.suggest_discrete_uniform('gamma', 0.5, 0.9999, 0.001)

    # Generate the Forest Management problem
    P, R, hole_states = create_frozen_lake_grid(grid_size, 0.10)
    hole_states = hole_states

    # Run Q-Learning
    ql_e_greedy = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=gamma, alpha=alpha, epsilon=0.1, epsilon_decay=0.99, n_iter=10000)
    ql_e_greedy.run()

    # Evaluate the performance (can be average reward, final reward, etc.)
    last_reward = ql_e_greedy.run_stats[-1]['Max V']
    return last_reward



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
P, R, hole_states = create_frozen_lake_grid(grid_size, 0.10)

# Q-Learning with Epsilon-Greedy Exploration
ql_ε_greedy = hiive.mdptoolbox.mdp.QLearning(P, R, gamma = study.best_params["gamma"], alpha = study.best_params["alpha"],  epsilon=0.1, epsilon_decay=0.99, n_iter=10000)
ql_ε_greedy.run()

results = {
    "gamma": study.best_params["gamma"],
    "alpha": study.best_params["alpha"],
    "numIterations": 10000,
    "policy": ql_ε_greedy.policy,
    "hole_states": hole_states
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

file_name = "fl_e_greedy_optimal_results.pickle"

import pickle
with open(file_name, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(results)
print("Best parameters:", study.best_params)
