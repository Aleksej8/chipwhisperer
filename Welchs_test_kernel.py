kernel = """
__kernel void t_test(__global const float* constant_trace,
                     __global const float* random_traces,
                     __global float* t_values,
                     const int trace_count,
                     const int samples)
{
    int gid = get_global_id(0);
        
        float constant_sum = 0.0f;
        float random_sum = 0.0f;
        for (int i = 0; i < trace_count; i++)
        {
            constant_sum += constant_trace[i * samples + gid];
            random_sum += random_traces[i * samples + gid];
        }
        float constant_mean = constant_sum / trace_count;
        float random_mean = random_sum / trace_count;
        
        float constant_var_sum = 0.0f;
        float random_var_sum = 0.0f;
        for (int i = 0; i < trace_count; i++)
        {
            float diff = constant_trace[i * samples + gid] - constant_mean;
            constant_var_sum += diff * diff;
            
            diff = random_traces[i * samples + gid] - random_mean;
            random_var_sum += diff * diff;
        }
        float constant_var = constant_var_sum / trace_count;
        float random_var = random_var_sum / trace_count;
        
        float t = (random_mean - constant_mean) / sqrt((random_var / trace_count) + (constant_var / trace_count));
        
        t_values[gid] = t;
}
"""

import pyopencl as cl
import numpy as np
import time
import matplotlib.pyplot as plt

platforms = cl.get_platforms()
devices = platforms[0].get_devices()
ctx = cl.Context([devices[0]])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

trace_array = np.load(r"trace_array.npy").astype(np.float32)
trace_array = trace_array[0:50]

constant_power_traces = trace_array[0:25]
random_power_traces = trace_array[25:50]
samples = len(constant_power_traces[1])
trace_count = len(constant_power_traces)

t_values = np.empty(samples, dtype=np.float32)
t_values_buf = cl.Buffer(ctx, mf.READ_WRITE, t_values.nbytes)

program = cl.Program(ctx, kernel).build()

start_time = time.time()

random_power_traces_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=random_power_traces)
constant_power_traces_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=constant_power_traces)

program.t_test(queue, (samples,), None, constant_power_traces_buf, random_power_traces_buf,
               t_values_buf, np.int32(trace_count), np.int32(samples))
cl.enqueue_copy(queue, t_values, t_values_buf)

end_time = time.time()

print("Executed time: ", end_time - start_time)
plt.plot(t_values)
plt.plot([4.5] * samples, color="red")
plt.plot([-4.5] * samples, color="red")
plt.show()