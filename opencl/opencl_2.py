kernel = """
    __constant uchar sbox[] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a, 
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e, 
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf, 
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16  
};

uint my_popcount(uint x) {
    uint count;
    for (count = 0; x; count++) {
        x &= x - 1;
    }
    return count;
}

float mean(float* X, int len) {
    float sum = 0;
    for (int i = 0; i < len; i++) {
        sum += X[i];
    }
    return sum / len;
}

float std_dev(float* X, float X_bar, int len) {
    float sum = 0;
    for (int i = 0; i < len; i++) {
        sum += (X[i] - X_bar) * (X[i] - X_bar);
    }
    return sqrt(sum); 
}


__kernel void attack(__global uchar* textin_array, __global double* trace_array, int trace_array_len, __global uchar* bestguess, __global float* cparefs)  {
    int bnum = get_global_id(0);
    int num_text_inputs = trace_array_len;
    double t_bar = 0.0, o_t = 0.0;
    double maxcpa = 0;
    uchar key_byte_guess = 0;
    float hws[50];
    double max_correlation = -1.0; 
    double correlation_array[5000];
    double o_t_array[5000];
    double t_bar_array[5000];
    
    // calc t_bar
    for (int j = 0; j < 5000; j++) {
        double sum = 0;
        for (int i = 0; i < 50; i++) {
            sum += trace_array[i * 5000 + j];
        }
        t_bar_array[j] = sum / 50;
    }

    // calc o_t
    for (int j = 0; j < 5000; j++) {
        double sum = 0;
        for (int i = 0; i < 50; i++) {
            double diff = trace_array[i * 5000 + j] - t_bar_array[j];
            sum += diff * diff;
        }
        o_t_array[j] = sqrt(sum);
    }
    
    for (int kguess = 0; kguess < 256; kguess++) { 
        for (int i = 0; i < num_text_inputs; i++) {
                hws[i] = __builtin_popcount(sbox[textin_array[i * 16 + bnum] ^ kguess]);
            }
            
        float hws_bar = mean(hws, num_text_inputs);
        float o_hws = std_dev(hws, hws_bar, num_text_inputs);
        
        for (int j = 0; j < 5000; j++) {
            double sum = 0;
            for (int i = 0; i < 50; i++) {
                double trace_diff = trace_array[i * 5000 + j] - t_bar_array[j];
                double hws_diff = hws[i] - hws_bar;
                sum += trace_diff * hws_diff;
            }
            correlation_array[j] = sum / (o_t_array[j] * o_hws);
            double correlation = fabs(correlation_array[j]);
            if (correlation > max_correlation) {
                max_correlation = correlation;
                key_byte_guess = kguess;
            }
        }
    }
    // Save the best guess and its correlation value in the output buffers
    bestguess[bnum] = key_byte_guess;
    cparefs[bnum] = max_correlation;
}
"""

import pyopencl as cl
from tqdm import tnrange
import numpy as np
import time

# get the start time
st = time.time()

trace_array = np.load(r"traces/lab4_2_traces.npy").astype(np.float64)
textin_array = np.load(r"traces/lab4_2_textin.npy")


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

textin_array_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=textin_array)
trace_array_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=trace_array)
bestguess_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=16 * np.dtype(np.uint8).itemsize)
cparefs_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=16 * np.dtype(np.float32).itemsize)

numtraces = 50 
numsubkeys = 16 

prg = cl.Program(ctx, kernel).build()

# Transfer input data to the device
cl.enqueue_copy(queue, textin_array_buf, textin_array)
cl.enqueue_copy(queue, trace_array_buf, trace_array)

# Call the OpenCL kernel
prg.attack(queue, (numsubkeys,), None, textin_array_buf, trace_array_buf, np.int32(numtraces), bestguess_buf, cparefs_buf)

# Retrieve the bestguess and cparefs results from the GPU
bestguess = np.empty(numsubkeys, dtype=np.uint8)
cparefs = np.empty(numsubkeys, dtype=np.float32)
cl.enqueue_copy(queue, bestguess, bestguess_buf)
cl.enqueue_copy(queue, cparefs, cparefs_buf)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

## Print the results for bestguess and cparefs
print("Best Key Guess:", end=" ")
for i, guess in enumerate(bestguess):
    print(f"{guess:02x}", end=" ")
print("\n")

print("[", end="")
for i, cpa_ref in enumerate(cparefs):
    if i != len(cparefs) - 1:
        print(f"{cpa_ref:.6f}, ", end="")
    else:
        print(f"{cpa_ref:.6f}", end="")
print("]")