// Updated kernel: multiply each element by 3 and add an increment based on the index
__kernel void my_kernel(__global float* buffer) {
    int id = get_global_id(0); // Get the global index
    buffer[id] = (buffer[id] * 3.0f) + (id * 0.1f); // Multiply by 3 and add a value based on the index
}
