import time
# from hiive.mdptoolbox.mdp import ValueIteration
from hiive.mdptoolbox.example import forest
import numpy as np
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp

np.random.seed(42)

# Define the optimal gamma and theta values for each state size
state_sizes = [10, 50, 100]
optimal_gamma = {
    10: 0.999,
    50: 0.999,  # Assuming you've decided on 0.99 after your analysis
    100: 0.999
}
optimal_theta = {
    10: 0,
    50: 0,
    100: 0
}

results = {}

file_name = '../forest_management_optimal_results/forest_management_optimal_results_pi.pickle'

# Run Value Iteration for each state size with the optimal gamma and theta
for size in state_sizes:
    print(size)
    # Get the transition and reward matrices for the forest management problem
    P, R = forest(S=size, r1=100, r2=10, p=0.1)

    # Start the timer
    start_time = time.time()

    # Create and run Value Iteration
    initial_policy = np.random.randint(2, size=size)
    vi = hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma=optimal_gamma[size], eval_type=optimal_theta[size], policy0 = initial_policy)
    vi.run()

    # Stop the timer
    elapsed_time = time.time() - start_time

    results[size] = {
        "gamma": optimal_gamma[size],
        "theta": optimal_theta[size],
        "numIterations": vi.iter,
        "policy": vi.policy,
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


    print('-' * 40)

# print(results)

import pickle
with open(file_name, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)