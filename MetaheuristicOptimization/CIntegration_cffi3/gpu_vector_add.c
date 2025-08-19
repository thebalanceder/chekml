#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define ARRAY_SIZE 1024 * 1024  // Increased array size
#define WORKGROUP_SIZE 64       // Workgroup size
#define REPEAT_COUNT 1000       // Repeat the kernel execution to keep GPU busy for longer

const char *kernelSource =
"__kernel void vec_add(__global const float *A, __global const float *B, __global float *C) {\n"
"    int id = get_global_id(0);\n"
"    C[id] = A[id] + B[id];\n"
"}\n";

int main() {
    float *A = (float*)malloc(sizeof(float) * ARRAY_SIZE);
    float *B = (float*)malloc(sizeof(float) * ARRAY_SIZE);
    float *C = (float*)malloc(sizeof(float) * ARRAY_SIZE);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    // Get platform
    cl_platform_id platforms[10];
    cl_uint num_platforms;
    clGetPlatformIDs(10, platforms, &num_platforms);
    cl_platform_id platform = platforms[1]; // Intel GPU

    // Get device
    cl_device_id devices[10];
    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 10, devices, &num_devices);
    cl_device_id device = devices[0];

    // Create context
    cl_int err;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    // Create queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

    // Create buffers
    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, A, &err);
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, B, &err);
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * ARRAY_SIZE, NULL, &err);

    // Compile kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Build log:\n%s\n", log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "vec_add", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buf);

    // Calculate global size (ensure it's large enough to cover all the elements)
    size_t global_size = ARRAY_SIZE;
    size_t local_size = WORKGROUP_SIZE;  // Size of workgroup (work items per workgroup)

    // Repeat kernel execution to keep GPU busy
    for (int i = 0; i < REPEAT_COUNT; i++) {
        // Run kernel with global and local size
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

        // Ensure kernel completes before moving to next iteration
        clFinish(queue);
    }

    // Read result
    err = clEnqueueReadBuffer(queue, c_buf, CL_TRUE, 0, sizeof(float) * ARRAY_SIZE, C, 0, NULL, NULL);

    // Wait for the kernel to finish
    clFinish(queue);

    // Verify result (only check first few elements to ensure computation is correct)
    for (int i = 0; i < 10; i++)
        printf("C[%d] = %.1f\n", i, C[i]);

    // Device info (to ensure you're using the correct GPU)
    char device_name[128];
    char device_vendor[128];
    cl_device_type device_type;

    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, NULL);
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);

    printf("Using device: %s\n", device_name);
    printf("Vendor: %s\n", device_vendor);
    printf("Device type: %s\n",
       (device_type & CL_DEVICE_TYPE_GPU) ? "GPU" :
       (device_type & CL_DEVICE_TYPE_CPU) ? "CPU" :
       "Other");

    // Cleanup
    clReleaseMemObject(a_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(c_buf);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(A); free(B); free(C);

    return 0;
}
