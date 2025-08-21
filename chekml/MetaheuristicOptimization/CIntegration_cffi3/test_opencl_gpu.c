#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define N 1024 * 1024 * 128  // 128 million floats (~512MB buffer)

const char *kernelSource =
"__kernel void stress_test(__global float *data) {\n"
"    int id = get_global_id(0);\n"
"    for (int i = 0; i < 1000; ++i) {\n"
"        data[id] = data[id] * 1.1f + 0.5f;\n"
"    }\n"
"}\n";

void check(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s failed: %d\n", msg, err);
        exit(1);
    }
}

int main() {
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem buffer;

    // Get Intel GPU platform
    err = clGetPlatformIDs(1, &platform, NULL);
    check(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    check(err, "clGetDeviceIDs");

    char name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("Using device: %s\n", name);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check(err, "clCreateContext");

    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    check(err, "clCreateCommandQueue");

    // Allocate and initialize host memory
    float *data = (float *)malloc(sizeof(float) * N);
    for (size_t i = 0; i < N; ++i) data[i] = (float)i;

    // Create OpenCL buffer
    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, data, &err);
    check(err, "clCreateBuffer");

    // Compile and create kernel
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    check(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build log:\n%s\n", log);
        free(log);
        exit(1);
    }

    kernel = clCreateKernel(program, "stress_test", &err);
    check(err, "clCreateKernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    check(err, "clSetKernelArg");

    size_t global_size = N;
    printf("Running stress test on %zu work items...\n", global_size);

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    check(err, "clEnqueueNDRangeKernel");

    clFinish(queue);
    printf("Execution complete.\n");

    // Cleanup
    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(data);

    return 0;
}

