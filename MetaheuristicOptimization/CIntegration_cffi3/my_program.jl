
using OpenCL

# Get platforms
platforms = OpenCL.platforms()

# Check if there are any platforms available
if length(platforms) == 0
    println("No OpenCL platforms found.")
    exit()
end

println("Available platforms: ", platforms)

# Select a platform (for example, the first one)
platform = platforms[1]

# Get devices for the selected platform
devices = OpenCL.devices(platform)

# Check if there are any devices available
if length(devices) == 0
    println("No devices found for the selected platform.")
    exit()
end

println("Available devices: ", devices)

# Select a device (for example, the first one, typically CPU or GPU)
device = devices[1]

# Create a context for the device
context = OpenCL.Context(device)

# Create a queue for the context
queue = OpenCL.Queue(context, device)

# Create a buffer on the device for input data (example: 1000 random numbers)
input_data = rand(Float32, 1000)
input_buffer = OpenCL.Buffer(context, input_data)

# Create a buffer for output data
output_data = zeros(Float32, 1000)
output_buffer = OpenCL.Buffer(context, output_data)

# Define OpenCL program source (kernel to square values)
source = """
__kernel void square(__global const float* input, __global float* output) {
    int i = get_global_id(0);
    output[i] = input[i] * input[i];
}
"""

# Create the OpenCL program from source code
program = OpenCL.Program(context, source)

# Build the program
build(program)

# Create the kernel from the program
kernel = OpenCL.Kernel(program, "square")

# Set kernel arguments
OpenCL.set_args(kernel, input_buffer, output_buffer)

# Execute the kernel (assuming one-dimensional array of size 1000)
OpenCL.enqueue_ndrange_kernel(queue, kernel, (1000,))

# Read back the output data from the device
OpenCL.read!(output_buffer)

# Print the first 10 squared values
println("Squared data (first 10 values): ", output_data[1:10])

