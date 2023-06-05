import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats

trace_array = np.load(r"trace_array.npy").astype(np.float32)
trace_array = trace_array[0:50]

constant_power_trace = trace_array[0]

random_power_traces = trace_array[1:]

def welch_test(constant_trace, random_traces):
    t_values = []
    for random_trace in random_traces:
        # Perform Welch's t-test on the power traces
        constant_mean = np.mean(constant_trace)
        random_mean = np.mean(random_trace)
        constant_std = np.std(constant_trace, ddof=1)
        random_std = np.std(constant_trace, ddof=1)

        t_value, _ = ttest_ind_from_stats(constant_mean, constant_std, len(constant_trace),
                                          random_mean, random_std, len(random_trace),
                                          equal_var=False)
        t_values.append(t_value)

    return t_values

# Perform Welch's t-test
t_values = welch_test(constant_power_trace, random_power_traces)

# Plot the p-values
plt.plot(t_values)
plt.axhline(0, color='r', linestyle='--', label='Zero Line')
plt.xlabel('Random Trace Index')
plt.ylabel('t-value') 
plt.title('Welch\'s t-test Results')
plt.legend()
plt.show()
