import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_name = 'fl_e_greedy_optimal_results.pickle'
with open(file_name, 'rb') as handle:
    results = pickle.load(handle)


max_v_data = []
mean_v_data = []
rewards_data = []
wallclock_data = []
state_sizes = []
iterations = []

# iterations = [range(len(max_v_data[0])), range(len(max_v_data[1])), range(len(max_v_data[2]))]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))  # Adjust the figsize as needed

state_sizes = [10]

# Iterate over each row (state size)
for i in range(1):
    print(i)
    size = state_sizes[i]
    # print("POLICY: ", results[state_sizes[i]]["policy"])
    # First column: Max V, Mean V, and Reward
    ax1 = axes[0]
    # ax1.plot(iterations[i], max_v_data[i], label='Max V')
    ax1.plot(range(results["numIterations"]), results["avgV"], label='Mean V')
    ax1.plot(range(results["numIterations"]), results["reward"], label='Reward')
    ax1.set_title(f'State Size {state_sizes[i]}: Metrics Over Iterations')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Value')
    ax1.legend()

    # Second column: Wall clock time
    ax2 = axes[1]
    ax2.plot(range(results["numIterations"]), results["wallClockTime"], label='Wall Clock Time', color='orange')
    ax2.set_title(f'State Size {state_sizes[i]}: Wall Clock Time Over Iterations')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Time (seconds)')
    ax2.legend()

# Adjust layout to prevent overlap
fig.tight_layout()
plt.show()