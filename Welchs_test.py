import numpy as np
import matplotlib.pyplot as plt

trace_array = np.load(r"trace_array.npy").astype(np.float32)
trace_array = trace_array[0:50]

constant_power_trace = trace_array[0:25]
random_power_traces = trace_array[25:50]

samples = len(constant_power_trace[1])

def mean(data):
    data[np.isnan(data)] = 0
    mean_data = data.mean(0)
    return mean_data


def var(data):
    data[np.isnan(data)] = 0
    var_data = data.var(0)
    return var_data


def welch_test(constant_trace, random_traces):
    constant_mean = mean(constant_trace)
    print(constant_mean)
    print(len(constant_mean))
    constant_var = var(constant_trace)
    print(constant_var)
    print(len(constant_var))
    random_mean = mean(random_traces)
    random_var = var(random_traces)
    trace_len = len(random_traces)

    t_value = (random_mean - constant_mean)/((random_var/trace_len + constant_var/trace_len) ** 0.5)

    return t_value


# Perform Welch's t-test
t_value = welch_test(constant_power_trace, random_power_traces)

# Plot the p-values
plt.plot(t_value)
plt.plot([4.5] * samples, color="red")
plt.plot([-4.5] * samples, color="red")
plt.show()
