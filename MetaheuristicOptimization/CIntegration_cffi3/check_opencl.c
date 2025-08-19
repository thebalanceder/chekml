// check_opencl.c
#include <stdio.h>
#include <CL/cl.h>

int main() {
    cl_uint num_platforms;
    cl_platform_id platforms[10];
    cl_int err = clGetPlatformIDs(10, platforms, &num_platforms);
    if (err != CL_SUCCESS) {
        printf("clGetPlatformIDs error: %d\n", err);
        return 1;
    }

    printf("Found %u OpenCL platforms.\n", num_platforms);

    for (cl_uint i = 0; i < num_platforms; ++i) {
        cl_device_id devices[10];
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 10, devices, &num_devices);
        if (err != CL_SUCCESS) {
            printf("  Platform %u: clGetDeviceIDs error: %d\n", i, err);
            continue;
        }
        printf("  Platform %u: Found %u device(s).\n", i, num_devices);
    }

    return 0;
}

