#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
    try {
        // Select the GPU device (specifically target the GPU)
        gpu_selector selector;

        // Create a SYCL queue for the selected device (GPU)
        queue q(selector);

        // Print out the selected device name
        std::cout << "Using device: " << q.get_device().get_info<info::device::name>() << std::endl;

        // Allocate memory for the result on the host
        int result = 0;

        // Create a buffer to store the result on the device
        buffer<int> result_buf(&result, range<1>(1));

        // Run the kernel on the selected device (GPU)
        q.submit([&](handler& h) {
            auto result_acc = result_buf.get_access<access::mode::write>(h);
            h.single_task<class simple_kernel>([=]() {
                result_acc[0] = 42; // Simple operation
            });
        }).wait(); // Ensure the kernel is executed before proceeding

        // After kernel execution, print the result
        std::cout << "Kernel execution complete. Result: " << result << std::endl;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

