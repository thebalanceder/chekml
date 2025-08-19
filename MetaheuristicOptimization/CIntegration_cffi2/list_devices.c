#include <CL/cl.h>
#include <stdio.h>

int main() {
    cl_platform_id platforms[10];
    cl_uint num_platforms;
    cl_int err;

    err = clGetPlatformIDs(10, platforms, &num_platforms);
    if (err != CL_SUCCESS) {
        printf("Error: clGetPlatformIDs failed (%d)\n", err);
        return 1;
    }

    printf("Found %u platform(s):\n", num_platforms);

    for (cl_uint i = 0; i < num_platforms; ++i) {
        char platform_name[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        printf("Platform %u: %s\n", i, platform_name);

        cl_device_id devices[10];
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 10, devices, &num_devices);
        if (err != CL_SUCCESS) {
            printf("  No devices found or clGetDeviceIDs failed (%d)\n", err);
            continue;
        }

        printf("  Found %u device(s):\n", num_devices);

        for (cl_uint j = 0; j < num_devices; ++j) {
            char device_name[128];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            printf("    Device %u: %s\n", j, device_name);
        }
    }

    return 0;
}

