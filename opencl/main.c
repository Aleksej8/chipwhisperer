#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#include <stdint.h>

#define MAX_SOURCE_SIZE (0x100000)
#define TRACE_ROWS 50
#define TRACE_COLS 5000
#define CHUNK_SIZE 1000



int main() {
    // Initialisierung von OpenCL
    cl_uint num_platforms = 0;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_int status = clGetPlatformIDs(0, NULL, &num_platforms);

    if (status != CL_SUCCESS) {
        printf("Fehler: Konnte Anzahl der verfügbaren OpenCL-Plattformen nicht abrufen.\n");
        exit(EXIT_FAILURE);
    }

    if (num_platforms > 0) {
        cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(num_platforms, platforms, NULL);

        // Nehmen Sie die erste verfügbare Plattform
        platform = platforms[0];
        free(platforms);
    }

    cl_context_properties properties[3];
    cl_context context;
    cl_command_queue command_queue;

    // Laden der NPY-Datei
    const char* npy_file_path = "C:/Users/Chris Funk/Studium/sem6/Projekt/textin_array.npy";
    FILE* textin_array;
    if (fopen_s(&textin_array, npy_file_path, "rb") != 0) {
        printf("Fehler: Konnte textin NPY-Datei nicht öffnen.\n");
        return 1;
    }

    // Laden der NPY-Datei für trace_array
    const char* npy_file_path3 = "C:/Users/Chris Funk/Studium/sem6/Projekt/trace_array.npy";
    FILE* trace_array;
    if (fopen_s(&trace_array, npy_file_path3, "rb") != 0) {
        printf("Fehler: Konnte trace_array NPY-Datei nicht öffnen.\n");
        return 1;
    }


    // Lesen Sie die ersten 6 Bytes des Headers und stellen Sie sicher, dass sie die magische Nummer enthalten
    char magic_number[6];
    fread(magic_number, sizeof(char), 6, textin_array);
    if (strncmp(magic_number, "\x93NUMPY", 6) != 0) {
        printf("Fehler: Ungültiges NPY-Dateiformat.\n");
        fclose(textin_array);
        exit(EXIT_FAILURE);
    }

    // Lesen Sie die nächsten 2 Bytes des Headers, die die Version der NPY-Datei enthalten
    char version[2];
    fread(version, sizeof(char), 2, textin_array);

    // Lesen Sie die nächsten 2 Bytes des Headers, die die Länge des verbleibenden Headers enthalten
    uint16_t header_length;
    fread(&header_length, sizeof(uint16_t), 1, textin_array);

    // Lesen Sie den Rest des Headers
    char* header = (char*)malloc(header_length * sizeof(char));
    fread(header, sizeof(char), header_length, textin_array);

    // Jetzt können Sie den Header-String ausgeben, um zu sehen, was er enthält
    printf("Header: %s\n", header);

    free(header);

    uint8_t data[50][16];

    // Jetzt, wo wir den Header gelesen haben, können wir das eigentliche Datenarray lesen.
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 16; ++j) {
            fread(&data[i][j], sizeof(uint8_t), 1, textin_array);
        }
    }

    fclose(textin_array);

    // Lesen Sie die ersten 6 Bytes des Headers und stellen Sie sicher, dass sie die magische Nummer enthalten
    char magic_number2[6];
    fread(magic_number2, sizeof(char), 6, trace_array);
    if (strncmp(magic_number2, "\x93NUMPY", 6) != 0) {
        printf("Fehler: Ungültiges NPY-Dateiformat.\n");
        fclose(trace_array);
        exit(EXIT_FAILURE);
    }

    // Lesen Sie die nächsten 2 Bytes des Headers, die die Version der NPY-Datei enthalten
    char version2[2];
    fread(version2, sizeof(char), 2, trace_array);

    // Lesen Sie die nächsten 2 Bytes des Headers, die die Länge des verbleibenden Headers enthalten
    uint16_t header_length2;
    fread(&header_length2, sizeof(uint16_t), 1, trace_array);

    // Lesen Sie den Rest des Headers
    char* header2 = (char*)malloc(header_length2 * sizeof(char));
    fread(header2, sizeof(char), header_length2, trace_array);

    // Jetzt können Sie den Header-String ausgeben, um zu sehen, was er enthält
    printf("Header: %s\n", header2);

    free(header2);

    

    double(*trace_data_double)[TRACE_COLS] = malloc(sizeof(double[TRACE_ROWS][TRACE_COLS]));
    float(*trace_data_float)[TRACE_COLS] = malloc(sizeof(float[TRACE_ROWS][TRACE_COLS]));

    // Jetzt, wo wir den Header gelesen haben, können wir das eigentliche Datenarray lesen.
    for (int i = 0; i < 50; ++i) {
        fread(trace_data_double[i], sizeof(double), 5000, trace_array);

        // Umwandeln in float
        for (int j = 0; j < 5000; ++j) {
            trace_data_float[i][j] = (float)trace_data_double[i][j];
        }
    }

    fclose(trace_array);

  
    
    
    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = (cl_context_properties)platform;
    properties[2] = 0;

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    if (status != CL_SUCCESS) {
        printf("Fehler: Konnte OpenCL-Gerät nicht abrufen.\n");
        exit(EXIT_FAILURE);
    }

    context = clCreateContext(properties, 1, &device, NULL, NULL, &status);
    command_queue = clCreateCommandQueue(context, device, 0, &status);

    // Laden und Kompilieren des OpenCL-Kernels
    
    const char* kernel_file_path = "C:/Users/Chris Funk/Studium/sem6/Projekt/kernel.cl";
    FILE* kernel_file;
    if (fopen_s(&kernel_file, kernel_file_path, "r") != 0) {
        printf("Fehler: Konnte kernel datei nicht öffnen.\n");
        return 1;
    }
   

    char* kernel_source = (char*)malloc(MAX_SOURCE_SIZE);
    size_t kernel_size = fread(kernel_source, 1, MAX_SOURCE_SIZE, kernel_file);

    fclose(kernel_file);

    int TRACE_LENGTH = 100; // Setzen Sie den gewünschten Wert
    int TRACE_COUNT = 50; // Setzen Sie den gewünschten Wert

    char build_options[100];
    sprintf_s(build_options, sizeof(build_options), "-D TRACE_LENGTH=%d -D TRACE_COUNT=%d", TRACE_LENGTH, TRACE_COUNT);



    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, (const size_t*)&kernel_size, &status);
    status = clBuildProgram(program, 1, &device, build_options, NULL, NULL);

    // Überprüfen Sie den Kompilierungsstatus und zeigen Sie etwaige Fehler an
    if (status != CL_SUCCESS) {
        cl_int build_status;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_int), &build_status, NULL);
        if (build_status != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char* log = (char*)malloc(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            printf("Fehler beim Kompilieren des OpenCL-Kernels:\n%s\n", log);
            free(log);
        }
    }

    // Jetzt erstellen wir das "calculate_hws"-Kernel
    cl_kernel kernel = clCreateKernel(program, "calculate_hws", &status);

    // Jetzt erstellen wir das "attack"-Kernel
    cl_kernel attack_kernel = clCreateKernel(program, "attack", &status);
    int numsubkeys = 16;
    
    // Erstellen Sie die Buffer für trace_array, bestguess und cparefs
    cl_mem trace_array_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 50  *5000* sizeof(float), trace_data_float, &status);
    
    cl_mem textin_array_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 50 * numsubkeys * sizeof(float), data, &status);
    cl_mem bestguess_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numsubkeys * sizeof(uint8_t), NULL, &status);
    cl_mem cparefs_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numsubkeys * sizeof(float), NULL, &status);
    cl_mem hws_array_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TRACE_COUNT * sizeof(float) * 256 * numsubkeys, NULL, &status);

    // Setzen Sie die Argumente des Kernels
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&textin_array_buf);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&hws_array_buf);
    if (status != CL_SUCCESS) {
        printf("Fehler: Konnte die Kernel-Argumente nicht setzen. Status: %d\n", status);
        exit(EXIT_FAILURE);
    }
    // Setzen Sie die Argumente des attack-Kernels
    status = clSetKernelArg(attack_kernel, 0, sizeof(cl_mem), (void*)&textin_array_buf);
    status |= clSetKernelArg(attack_kernel, 1, sizeof(cl_mem), (void*)&trace_array_buf);
    status |= clSetKernelArg(attack_kernel, 2, sizeof(cl_mem), (void*)&bestguess_buf);
    status |= clSetKernelArg(attack_kernel, 3, sizeof(cl_mem), (void*)&cparefs_buf);
    status |= clSetKernelArg(attack_kernel, 4, sizeof(cl_mem), (void*)&hws_array_buf);

    // Starten Sie die Ausführung des Kernels
    size_t global_size = numsubkeys;
    status = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Fehler: hw calc Kernel konnte nicht ausgeführt werden. Status: %d\n", status);
        exit(EXIT_FAILURE);
    }
    clFlush(command_queue);
    clFinish(command_queue);
    // Starten Sie die Ausführung des attack-Kernels
    status = clEnqueueNDRangeKernel(command_queue, attack_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Fehler: atack Kernel konnte nicht ausgeführt werden. Status: %d\n", status);
        exit(EXIT_FAILURE);
    }
    clFlush(command_queue);
    clFinish(command_queue);
    // Ergebnisse vom OpenCL-Gerät abrufen
    float* hws_array = (float*)malloc(TRACE_COUNT * sizeof(float) * 256 * numsubkeys);
    status = clEnqueueReadBuffer(command_queue, hws_array_buf, CL_TRUE, 0, TRACE_COUNT * sizeof(float) * 256 * numsubkeys, hws_array, 0, NULL, NULL);

    // Ergebnisse vom OpenCL-Gerät abrufen
    uint8_t* bestguess = (uint8_t*)malloc(numsubkeys * sizeof(uint8_t));
    status = clEnqueueReadBuffer(command_queue, bestguess_buf, CL_TRUE, 0, numsubkeys * sizeof(uint8_t), bestguess, 0, NULL, NULL);

    float* cparefs = (float*)malloc(numsubkeys * sizeof(float));
    status = clEnqueueReadBuffer(command_queue, cparefs_buf, CL_TRUE, 0, numsubkeys * sizeof(float), cparefs, 0, NULL, NULL);

    printf("Best Key Guess: ");
    for (int i = 0; i < numsubkeys; i++) {
        printf("%02x ", bestguess[i]);
    }
    printf("\n");

    printf("[");
    for (int i = 0; i < numsubkeys; i++) {
        if (i != numsubkeys - 1) {
            printf("%.6f, ", cparefs[i]);
        }
        else {
            printf("%.6f", cparefs[i]);
        }
    }
    printf("]\n");

    // Aufräumen
    free(trace_data_double);
    free(trace_data_float);
    free(kernel_source);
    free(bestguess);
    free(cparefs);
    clReleaseMemObject(textin_array_buf);
    clReleaseMemObject(trace_array_buf);  
    clReleaseMemObject(bestguess_buf);
    clReleaseMemObject(cparefs_buf);
    clReleaseMemObject(hws_array_buf);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    printf("Programm beendet.\n");
    return 0;
}
