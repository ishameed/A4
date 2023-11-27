import time
from hiive.mdptoolbox.mdp import ValueIteration
from hiive.mdptoolbox.example import forest
import numpy as np

np.random.seed(42)

# Define the optimal gamma and theta values for each state size
state_sizes = [20, 25, 30]
optimal_gamma = {
    20: 0.792,
    25: 0.9670000000000001,  # Assuming you've decided on 0.99 after your analysis
    30: 0.855
}
optimal_theta = {
    20: 0.00051,
    25: 0.00030000000000000003,
    30: 0.00064
}

results = {}

file_name = '../frozen_lake_optimal_results/frozen_lake_optimal_results.pickle'

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

hole_states = []
# Run Value Iteration for each state size with the optimal gamma and theta
for size in state_sizes:
    print(size)
    # Get the transition and reward matrices for the forest management problem
    P, R, hole_states = create_frozen_lake_grid(grid_size=size, hole_probability=0.1)
    hole_states = hole_states

    # Start the timer
    start_time = time.time()

    # Create and run Value Iteration
    vi = ValueIteration(P, R, gamma=optimal_gamma[size], epsilon=optimal_theta[size])
    vi.run()

    # Stop the timer
    elapsed_time = time.time() - start_time

    results[size] = {
        "gamma": optimal_gamma[size],
        "theta": optimal_theta[size],
        "numIterations": vi.iter,
        "policy": vi.policy,
        "hole_states": hole_states
    }

    results[size]["maxV"] = []
    results[size]["avgV"] = []
    results[size]["reward"] = []
    results[size]["wallClockTime"] = []

    for iteration in vi.run_stats:
        results[size]["maxV"].append(iteration["Max V"])
        results[size]["avgV"].append(iteration["Mean V"])
        results[size]["reward"].append(iteration["Reward"])
        results[size]["wallClockTime"].append(iteration["Time"])

    # print("size: ", size)
    # print("REWARD: ", results[size]["reward"][:10])
    # print("max: ", results[size]["maxV"][:10])
    # print("mean: ", results[size]["avgV"][:10])

    print('-' * 40)

# print(results)

import pickle
with open(file_name, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


