
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_name = '../frozen_lake_optimal_results/frozen_lake_optimal_results_pi.pickle'
with open(file_name, 'rb') as handle:
    results = pickle.load(handle)


def visualize_policy(policy, grid_size, hole_states):
    """
    Visualizes the policy for the Frozen Lake problem.

    :param policy: A list or array of actions, where each action is an integer.
    :param grid_size: The size of one side of the grid (e.g., 30 for a 30x30 grid).
    """
    # Define symbols for each action (assuming 0: left, 1: down, 2: right, 3: up)
    action_symbols = {0: '<', 1: 'v', 2: '>', 3: '^'}

    # Create a grid to represent the policy
    policy_grid = np.array([action_symbols[action] for action in policy]).reshape((grid_size, grid_size))
    hole_positions = [(state // grid_size, state % grid_size) for state in hole_states]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.matshow(np.zeros((grid_size, grid_size)), cmap='viridis')

    for (i, j), action in np.ndenumerate(policy_grid):
        if (i, j) in hole_positions:
            continue
        if i == 0 and j == 0:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='green', alpha=0.3))  # Start
        elif i == grid_size - 1 and j == grid_size - 1:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='red', alpha=0.3))  # Goal
        ax.text(j, i, action, ha='center', va='center', color='white')

    plt.xticks(np.arange(grid_size))
    plt.yticks(np.arange(grid_size))
    plt.title('Frozen Lake Policy Visualization')
    plt.show()

# Example Usage
grid_size = 25
policy = results[grid_size]["policy"] # Example policy
hole_states = results[grid_size]["hole_states"]
print("pOLICY: ", policy)
print("HOLE STATES: ", results[grid_size]["hole_states"])
visualize_policy(policy, grid_size, hole_states)