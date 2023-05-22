#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_SOURCE_SIZE (0x100000)
#define NUM_CHUNK_SIZES 7
#define NUM_SUBKEYS 16
const char *kernel_source = R"(
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


__kernel void attack(__global uchar* textin_array, __global float* trace_array, int trace_array_len, __global uchar* bestguess, __global float* cparefs)  {
    int bnum = get_global_id(0);
    int num_text_inputs = trace_array_len;
    float t_bar = 0.0, o_t = 0.0;
    float maxcpa = 0;
    uchar key_byte_guess = 0;
    float hws[50];
    float max_correlation = -1.0; 
    float correlation_array[TRACE_LENGTH];
    float o_t_array[TRACE_LENGTH];
    float t_bar_array[TRACE_LENGTH];
    
    // calc t_bar
    for (int j = 0; j < TRACE_LENGTH; j++) {
        float sum = 0;
        for (int i = 0; i < num_text_inputs; i++) {
            sum += trace_array[i * TRACE_LENGTH + j];
        }
        t_bar_array[j] = sum / num_text_inputs;
    }
    
    // calc o_t
    for (int j = 0; j < TRACE_LENGTH; j++) {
        float sum = 0;
        for (int i = 0; i < num_text_inputs; i++) {
            float diff = trace_array[i * TRACE_LENGTH + j] - t_bar_array[j];
            sum += diff * diff;
        }
        o_t_array[j] = sqrt(sum);
    }
    
    for (int kguess = 0; kguess < 256; kguess++) { 
        for (int i = 0; i < num_text_inputs; i++) {
                hws[i] = my_popcount(sbox[textin_array[i * 16 + bnum] ^ kguess]);
            }
            
        float hws_bar = mean(hws, num_text_inputs);
        float o_hws = std_dev(hws, hws_bar, num_text_inputs);
        
        for (int j = 0; j < TRACE_LENGTH; j++) {
            float sum = 0;
            for (int i = 0; i < num_text_inputs; i++) {
                float trace_diff = trace_array[i * TRACE_LENGTH + j] - t_bar_array[j];
                float hws_diff = hws[i] - hws_bar;
                sum += trace_diff * hws_diff;
            }
            correlation_array[j] = sum / (o_t_array[j] * o_hws);
            float correlation = fabs(correlation_array[j]);
            if (correlation > max_correlation) {
                max_correlation = correlation;
                key_byte_guess = kguess;
            }
        }
    }
    // Save the best guess and its correlation value in the output buffers
    bestguess[bnum] = key_byte_guess;
    cparefs[bnum] = max_correlation;

)";
int main() {
	FILE* file = fopen("trace_array.txt", "r");
	if (file == NULL) {
		printf("Cannot open file\n");
		exit(0);
	}

	int count = 0;
	float value;

	while (fscanf(file, "%f", &value) != EOF) {
		// Hier sollten Sie den gelesenen Wert irgendwo speichern
		// Zum Beispiel in einem Array, das groß genug ist, um alle Werte zu speichern.
		data[count] = value;
		count++;
	}

	fclose(file);

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    cl_mem textin_array_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_uint) * TEXTIN_SIZE, NULL, &ret);
    cl_mem bestguess_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * NUM_SUBKEYS, NULL, &ret);
    cl_mem cparefs_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * NUM_SUBKEYS, NULL, &ret);

    // Laden und Bauen des OpenCL-Programms 
   
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    
    // Erstellen des OpenCL-Kernels
    cl_kernel kernel = clCreateKernel(program, "attack", &ret);
	
    int chunk_sizes[NUM_CHUNK_SIZES] = {10, 50, 100, 500, 1000, 2500, 5000};
    float execution_times[NUM_CHUNK_SIZES];
    
    for (int i = 0; i < NUM_CHUNK_SIZES; ++i) {
        int chunk_size = chunk_sizes[i];
        printf("Processing chunk size %d\n", chunk_size);

		// Überprüfen der Teilbarkeit
        if (count % chunk_size != 0) {
            printf("The number of columns in trace_array must be divisible by the chunk size.");
            exit(1);
        }

        clock_t start_time = clock();

        for (int chunk_index = 0; chunk_index < num_trace_array_chunks; ++chunk_index) {
            cl_mem trace_array_chunk_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * chunk_size, NULL, &ret);

             // Aufteilen von trace_array
            float* trace_array_chunk = data + chunk_index * chunk_size;

            cl_mem trace_array_chunk_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * chunk_size, trace_array_chunk, &ret);

            // Ausführen des OpenCL-Kernels
            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&textin_array_mem_obj);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&trace_array_chunk_mem_obj);
            // Setzen Sie hier die restlichen Kernel-Argumente
            size_t global_item_size = chunk_size;
            size_t local_item_size = 1;
            ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

            // Abrufen der Ergebnisse
            cl_uint* bestguess = (cl_uint*)malloc(sizeof(cl_uint) * NUM_SUBKEYS);
            cl_float* cparefs = (cl_float*)malloc(sizeof(cl_float) * NUM_SUBKEYS);
            ret = clEnqueueReadBuffer(command_queue, bestguess_mem_obj, CL_TRUE, 0, sizeof(cl_uint) * NUM_SUBKEYS, bestguess, 0, NULL, NULL);
            ret = clEnqueueReadBuffer(command_queue, cparefs_mem_obj, CL_TRUE, 0, sizeof(cl_float) * NUM_SUBKEYS, cparefs, 0, NULL, NULL);

			// Verarbeiten Sie hier die Ergebnisse
            // ...
            clReleaseMemObject(trace_array_chunk_mem_obj);
			
			free(bestguess);
            free(cparefs);
        }

        clock_t end_time = clock();
        execution_times[i] = ((float) (end_time - start_time)) / CLOCKS_PER_SEC;

        // Code zum Bestimmen der besten Indices und Werte fehlt hier
    }

    // Aufräumen
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
	
	free(source_str);
    free(data);

    return 0;
}
