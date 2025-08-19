#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <omp.h>
#include <string>
#include <iostream>

namespace py = pybind11;

// OpenCL kernel source code
const std::string kernel_source = R"(
__kernel void compute_distance_matrix(__global const double* input, __global double* output, int n) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < n && j < n) {
        output[i * n + j] = fabs(input[i] - input[j]);
    }
}

__kernel void center_matrix(__global double* matrix, __global const double* row_mean, 
                          __global const double* col_mean, double mean, int n) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < n && j < n) {
        matrix[i * n + j] = matrix[i * n + j] - row_mean[i] - col_mean[j] + mean;
    }
}

__kernel void gaussian_kernel_matrix(__global const double* dist, double sigma, __global double* output, int n) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < n && j < n) {
        output[i * n + j] = exp(-dist[i * n + j] * dist[i * n + j] / (2 * sigma * sigma + 1e-10));
    }
}

__kernel void reduction_sum(__global const double* input, __global double* output, int n, __local double* sdata) {
    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);
    sdata[tid] = (i < n) ? input[i] : 0.0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        output[get_group_id(0)] = sdata[0];
    }
}
)";

// Global OpenCL objects
cl::Context context;
cl::CommandQueue queue;
cl::Program program;
bool opencl_initialized = false;

void initialize_opencl() {
    if (opencl_initialized) return;

    try {
        // Get platforms
        std::vector<cl::Platform> platforms;
        cl_int err = cl::Platform::get(&platforms);
        if (err != CL_SUCCESS || platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found, error: " + std::to_string(err));
        }

        // Prefer Intel platform
        cl::Platform selected_platform = platforms[0];
        for (const auto& platform : platforms) {
            std::string name = platform.getInfo<CL_PLATFORM_NAME>();
            if (name.find("Intel") != std::string::npos) {
                selected_platform = platform;
                break;
            }
        }

        // Get devices
        std::vector<cl::Device> devices;
        err = selected_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (err != CL_SUCCESS || devices.empty()) {
            err = selected_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            if (err != CL_SUCCESS || devices.empty()) {
                throw std::runtime_error("No OpenCL devices found, error: " + std::to_string(err));
            }
        }

        // Prefer Intel GPU device
        cl::Device selected_device = devices[0];
        for (const auto& device : devices) {
            std::string name = device.getInfo<CL_DEVICE_NAME>();
            if (name.find("Intel") != std::string::npos && name.find("Graphics") != std::string::npos) {
                selected_device = device;
                break;
            }
        }

        // Create context and queue
        context = cl::Context(selected_device);
        queue = cl::CommandQueue(context, selected_device);

        // Create and build program
        program = cl::Program(context, kernel_source);
        err = program.build({selected_device});
        if (err != CL_SUCCESS) {
            std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device);
            throw std::runtime_error("OpenCL build error: " + build_log);
        }

        opencl_initialized = true;
    } catch (const std::exception& e) {
        std::cerr << "OpenCL initialization failed: " << e.what() << std::endl;
        throw;
    }
}

double compute_distance_correlation_cpu(py::array_t<double> x, py::array_t<double> y) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    if (x_buf.ndim != 1 || y_buf.ndim != 1 || x_buf.shape[0] != y_buf.shape[0]) {
        throw std::runtime_error("Input arrays must be 1D and of equal length");
    }
    int n = x_buf.shape[0];
    if (n < 2) return 0.0;

    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);

    std::vector<double> A(n * n), B(n * n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = fabs(x_ptr[i] - x_ptr[j]);
            B[i * n + j] = fabs(y_ptr[i] - y_ptr[j]);
        }
    }

    double A_mean = 0.0, B_mean = 0.0;
    #pragma omp parallel for reduction(+:A_mean,B_mean)
    for (int i = 0; i < n * n; i++) {
        A_mean += A[i];
        B_mean += B[i];
    }
    A_mean /= (n * n);
    B_mean /= (n * n);

    std::vector<double> row_mean(n, 0.0), col_mean(n, 0.0);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            row_mean[i] += A[i * n + j];
            col_mean[i] += A[j * n + i];
        }
        row_mean[i] /= n;
        col_mean[i] /= n;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = A[i * n + j] - row_mean[i] - col_mean[j] + A_mean;
            B[i * n + j] = B[i * n + j] - row_mean[i] - col_mean[j] + B_mean;
        }
    }

    double dCov = 0.0, dVar_x = 0.0, dVar_y = 0.0;
    #pragma omp parallel for reduction(+:dCov,dVar_x,dVar_y)
    for (int i = 0; i < n * n; i++) {
        dCov += A[i] * B[i];
        dVar_x += A[i] * A[i];
        dVar_y += B[i] * B[i];
    }
    dCov = sqrt(fabs(dCov / (n * n)));
    dVar_x = sqrt(fabs(dVar_x / (n * n)));
    dVar_y = sqrt(fabs(dVar_y / (n * n)));

    return (dVar_x > 0 && dVar_y > 0) ? dCov / sqrt(dVar_x * dVar_y) : 0.0;
}

double compute_distance_correlation(py::array_t<double> x, py::array_t<double> y) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    if (x_buf.ndim != 1 || y_buf.ndim != 1 || x_buf.shape[0] != y_buf.shape[0]) {
        throw std::runtime_error("Input arrays must be 1D and of equal length");
    }
    int n = x_buf.shape[0];
    if (n < 2) return 0.0;

    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);

    try {
        initialize_opencl();

        // Create buffers
        cl::Buffer d_x(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(double), x_ptr);
        cl::Buffer d_y(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(double), y_ptr);
        cl::Buffer d_A(context, CL_MEM_READ_WRITE, n * n * sizeof(double));
        cl::Buffer d_B(context, CL_MEM_READ_WRITE, n * n * sizeof(double));
        cl::Buffer d_row_mean(context, CL_MEM_READ_WRITE, n * sizeof(double));
        cl::Buffer d_col_mean(context, CL_MEM_READ_WRITE, n * sizeof(double));
        cl::Buffer d_temp(context, CL_MEM_READ_WRITE, n * sizeof(double));

        // Compute distance matrices
        cl::Kernel distance_kernel(program, "compute_distance_matrix");
        distance_kernel.setArg(0, d_x);
        distance_kernel.setArg(1, d_A);
        distance_kernel.setArg(2, n);
        cl_int err = queue.enqueueNDRangeKernel(distance_kernel, cl::NullRange, cl::NDRange(n, n), cl::NullRange);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue distance kernel for A: " + std::to_string(err));
        }

        distance_kernel.setArg(0, d_y);
        distance_kernel.setArg(1, d_B);
        err = queue.enqueueNDRangeKernel(distance_kernel, cl::NullRange, cl::NDRange(n, n), cl::NullRange);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue distance kernel for B: " + std::to_string(err));
        }
        queue.finish();

        // Compute means
        double A_mean = 0.0, B_mean = 0.0;
        cl::Kernel sum_kernel(program, "reduction_sum");
        sum_kernel.setArg(0, d_A);
        sum_kernel.setArg(1, d_temp);
        sum_kernel.setArg(2, n * n);
        sum_kernel.setArg(3, sizeof(double) * 256, nullptr);
        err = queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(n * n), cl::NDRange(256));
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue sum kernel for A: " + std::to_string(err));
        }
        queue.finish();

        std::vector<double> temp(n);
        err = queue.enqueueReadBuffer(d_temp, CL_TRUE, 0, n * sizeof(double), temp.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read temp buffer for A: " + std::to_string(err));
        }
        for (double val : temp) A_mean += val;
        A_mean /= (n * n);

        sum_kernel.setArg(0, d_B);
        err = queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(n * n), cl::NDRange(256));
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue sum kernel for B: " + std::to_string(err));
        }
        queue.finish();
        err = queue.enqueueReadBuffer(d_temp, CL_TRUE, 0, n * sizeof(double), temp.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read temp buffer for B: " + std::to_string(err));
        }
        for (double val : temp) B_mean += val;
        B_mean /= (n * n);

        // Compute row and column means
        std::vector<double> row_mean(n, 0.0), col_mean(n, 0.0);
        std::vector<double> A(n * n), B(n * n);
        err = queue.enqueueReadBuffer(d_A, CL_TRUE, 0, n * n * sizeof(double), A.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read A buffer: " + std::to_string(err));
        }
        err = queue.enqueueReadBuffer(d_B, CL_TRUE, 0, n * n * sizeof(double), B.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read B buffer: " + std::to_string(err));
        }

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double row_sum = 0.0, col_sum = 0.0;
            for (int j = 0; j < n; j++) {
                row_sum += A[i * n + j];
                col_sum += A[j * n + i];
            }
            row_mean[i] = row_sum / n;
            col_mean[i] = col_sum / n;
        }

        err = queue.enqueueWriteBuffer(d_row_mean, CL_TRUE, 0, n * sizeof(double), row_mean.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to write row_mean buffer: " + std::to_string(err));
        }
        err = queue.enqueueWriteBuffer(d_col_mean, CL_TRUE, 0, n * sizeof(double), col_mean.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to write col_mean buffer: " + std::to_string(err));
        }

        // Center matrices
        cl::Kernel center_kernel(program, "center_matrix");
        center_kernel.setArg(0, d_A);
        center_kernel.setArg(1, d_row_mean);
        center_kernel.setArg(2, d_col_mean);
        center_kernel.setArg(3, A_mean);
        center_kernel.setArg(4, n);
        err = queue.enqueueNDRangeKernel(center_kernel, cl::NullRange, cl::NDRange(n, n), cl::NullRange);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue center kernel for A: " + std::to_string(err));
        }

        center_kernel.setArg(0, d_B);
        center_kernel.setArg(3, B_mean);
        err = queue.enqueueNDRangeKernel(center_kernel, cl::NullRange, cl::NDRange(n, n), cl::NullRange);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue center kernel for B: " + std::to_string(err));
        }
        queue.finish();

        // Compute dCov
        double dCov = 0.0;
        sum_kernel.setArg(0, d_A);
        err = queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(n * n), cl::NDRange(256));
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue sum kernel for dCov: " + std::to_string(err));
        }
        queue.finish();
        err = queue.enqueueReadBuffer(d_temp, CL_TRUE, 0, n * sizeof(double), temp.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read temp buffer for dCov: " + std::to_string(err));
        }
        for (double val : temp) dCov += val;
        dCov = sqrt(fabs(dCov / (n * n)));

        // Compute dVar_x
        double dVar_x = 0.0;
        sum_kernel.setArg(0, d_A);
        err = queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(n * n), cl::NDRange(256));
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue sum kernel for dVar_x: " + std::to_string(err));
        }
        queue.finish();
        err = queue.enqueueReadBuffer(d_temp, CL_TRUE, 0, n * sizeof(double), temp.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read temp buffer for dVar_x: " + std::to_string(err));
        }
        for (double val : temp) dVar_x += val * val;
        dVar_x = sqrt(fabs(dVar_x / (n * n)));

        // Compute dVar_y
        double dVar_y = 0.0;
        sum_kernel.setArg(0, d_B);
        err = queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(n * n), cl::NDRange(256));
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue sum kernel for dVar_y: " + std::to_string(err));
        }
        queue.finish();
        err = queue.enqueueReadBuffer(d_temp, CL_TRUE, 0, n * sizeof(double), temp.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read temp buffer for dVar_y: " + std::to_string(err));
        }
        for (double val : temp) dVar_y += val * val;
        dVar_y = sqrt(fabs(dVar_y / (n * n)));

        double dCor = (dVar_x > 0 && dVar_y > 0) ? dCov / sqrt(dVar_x * dVar_y) : 0.0;
        return dCor;

    } catch (const std::exception& e) {
        std::cerr << "OpenCL failed, falling back to CPU: " << e.what() << std::endl;
        return compute_distance_correlation_cpu(x, y);
    }
}

void compute_quantiles(const std::vector<double>& data, int n, int q, std::vector<double>& quantiles) {
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    #pragma omp parallel for
    for (int i = 0; i < q; i++) {
        double idx = i * (n - 1.0) / (q - 1);
        int lower = static_cast<int>(idx);
        double frac = idx - lower;
        if (lower + 1 < n) {
            quantiles[i] = sorted[lower] * (1 - frac) + sorted[lower + 1] * frac;
        } else {
            quantiles[i] = sorted[lower];
        }
    }
}

double compute_mic(py::array_t<double> x, py::array_t<double> y, int max_bins) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    if (x_buf.ndim != 1 || y_buf.ndim != 1 || x_buf.shape[0] != y_buf.shape[0]) {
        throw std::runtime_error("Input arrays must be 1D and of equal length");
    }
    int n = x_buf.shape[0];
    if (n < 2) return 0.0;

    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);
    std::vector<double> x_vec(x_ptr, x_ptr + n);
    std::vector<double> y_vec(y_ptr, y_ptr + n);

    max_bins = std::min(max_bins, static_cast<int>(ceil(pow(n, 0.6))));
    double mic = 0.0;

    #pragma omp parallel
    {
        double local_mic = 0.0;
        #pragma omp for schedule(dynamic)
        for (int i = 2; i <= max_bins; i++) {
            for (int j = 2; j <= max_bins; j++) {
                if (i * j > max_bins * max_bins) continue;

                std::vector<double> x_quantiles(i), y_quantiles(j);
                compute_quantiles(x_vec, n, i, x_quantiles);
                compute_quantiles(y_vec, n, j, y_quantiles);

                std::vector<int> x_bins(n, 0), y_bins(n, 0);
                #pragma omp parallel for
                for (int k = 0; k < n; k++) {
                    for (int l = 0; l < i - 1; l++) {
                        if (x_vec[k] <= x_quantiles[l + 1]) {
                            x_bins[k] = l;
                            break;
                        }
                    }
                    if (x_vec[k] > x_quantiles[i - 1]) x_bins[k] = i - 1;
                    for (int l = 0; l < j - 1; l++) {
                        if (y_vec[k] <= y_quantiles[l + 1]) {
                            y_bins[k] = l;
                            break;
                        }
                    }
                    if (y_vec[k] > y_quantiles[j - 1]) y_bins[k] = j - 1;
                }

                std::vector<double> joint_hist(i * j, 0.0);
                #pragma omp parallel for
                for (int k = 0; k < n; k++) {
                    #pragma omp atomic
                    joint_hist[x_bins[k] * j + y_bins[k]] += 1.0 / n;
                }

                std::vector<double> x_hist(i, 0.0), y_hist(j, 0.0);
                for (int k = 0; k < i; k++) {
                    for (int l = 0; l < j; l++) {
                        x_hist[k] += joint_hist[k * j + l];
                        y_hist[l] += joint_hist[k * j + l];
                    }
                }

                double mi = 0.0;
                #pragma omp parallel for reduction(+:mi)
                for (int k = 0; k < i; k++) {
                    for (int l = 0; l < j; l++) {
                        if (joint_hist[k * j + l] > 0) {
                            mi += joint_hist[k * j + l] * log2(joint_hist[k * j + l] / (x_hist[k] * y_hist[l] + 1e-10));
                        }
                    }
                }

                double norm_mi = mi / log2(static_cast<double>(std::min(i, j)));
                if (norm_mi > local_mic) local_mic = norm_mi;
            }
        }
        #pragma omp critical
        if (local_mic > mic) mic = local_mic;
    }

    return mic;
}

void rankdata(const std::vector<double>& data, std::vector<double>& ranks, int n) {
    std::vector<std::pair<double, int>> pairs(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        pairs[i] = {data[i], i};
    }

    std::sort(pairs.begin(), pairs.end());

    int i = 0;
    while (i < n) {
        int start = i;
        double sum_ranks = 0;
        int count = 0;
        while (i < n && pairs[i].first == pairs[start].first) {
            sum_ranks += i + 1;
            count++;
            i++;
        }
        double avg_rank = sum_ranks / count;
        #pragma omp parallel for
        for (int j = start; j < i; j++) {
            ranks[pairs[j].second] = avg_rank;
        }
    }
}

double compute_copula_measure(py::array_t<double> x, py::array_t<double> y) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    if (x_buf.ndim != 1 || y_buf.ndim != 1 || x_buf.shape[0] != y_buf.shape[0]) {
        throw std::runtime_error("Input arrays must be 1D and of equal length");
    }
    int n = x_buf.shape[0];

    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);
    std::vector<double> x_vec(x_ptr, x_ptr + n);
    std::vector<double> y_vec(y_ptr, y_ptr + n);
    std::vector<double> x_ranks(n), y_ranks(n), x_norm(n), y_norm(n);

    rankdata(x_vec, x_ranks, n);
    rankdata(y_vec, y_ranks, n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x_norm[i] = x_ranks[i] / (n + 1);
        y_norm[i] = y_ranks[i] / (n + 1);
    }

    double mean_x = 0.0, mean_y = 0.0;
    #pragma omp parallel for reduction(+:mean_x,mean_y)
    for (int i = 0; i < n; i++) {
        mean_x += x_norm[i];
        mean_y += y_norm[i];
    }
    mean_x /= n;
    mean_y /= n;

    double cov = 0.0, var_x = 0.0, var_y = 0.0;
    #pragma omp parallel for reduction(+:cov,var_x,var_y)
    for (int i = 0; i < n; i++) {
        cov += (x_norm[i] - mean_x) * (y_norm[i] - mean_y);
        var_x += (x_norm[i] - mean_x) * (x_norm[i] - mean_x);
        var_y += (y_norm[i] - mean_y) * (y_norm[i] - mean_y);
    }
    cov /= n;
    var_x /= n;
    var_y /= n;

    double corr = (var_x > 0 && var_y > 0) ? cov / sqrt(var_x * var_y) : 0.0;
    return fabs(corr);
}

double compute_hsic_cpu(py::array_t<double> x, py::array_t<double> y) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    if (x_buf.ndim != 1 || y_buf.ndim != 1 || x_buf.shape[0] != y_buf.shape[0]) {
        throw std::runtime_error("Input arrays must be 1D and of equal length");
    }
    int n = x_buf.shape[0];
    if (n < 2) return 0.0;

    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);

    std::vector<double> dist_x(n * n), dist_y(n * n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dist_x[i * n + j] = fabs(x_ptr[i] - x_ptr[j]);
            dist_y[i * n + j] = fabs(y_ptr[i] - y_ptr[j]);
        }
    }

    std::vector<double> non_zero_dist_x, non_zero_dist_y;
    for (double val : dist_x) if (val > 0) non_zero_dist_x.push_back(val);
    for (double val : dist_y) if (val > 0) non_zero_dist_y.push_back(val);
    double sigma_x = 1.0, sigma_y = 1.0;
    if (!non_zero_dist_x.empty()) {
        std::sort(non_zero_dist_x.begin(), non_zero_dist_x.end());
        sigma_x = non_zero_dist_x[non_zero_dist_x.size() / 2];
    }
    if (!non_zero_dist_y.empty()) {
        std::sort(non_zero_dist_y.begin(), non_zero_dist_y.end());
        sigma_y = non_zero_dist_y[non_zero_dist_y.size() / 2];
    }

    std::vector<double> K(n * n), L(n * n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            K[i * n + j] = exp(-dist_x[i * n + j] * dist_x[i * n + j] / (2 * sigma_x * sigma_x + 1e-10));
            L[i * n + j] = exp(-dist_y[i * n + j] * dist_y[i * n + j] / (2 * sigma_y * sigma_y + 1e-10));
        }
    }

    std::vector<double> H(n * n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            H[i * n + j] = (i == j) ? 1.0 - 1.0 / n : -1.0 / n;
        }
    }

    std::vector<double> HK(n * n), HL(n * n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum_HK = 0.0, sum_HL = 0.0;
            for (int k = 0; k < n; k++) {
                sum_HK += H[i * n + k] * K[k * n + j];
                sum_HL += H[i * n + k] * L[k * n + j];
            }
            HK[i * n + j] = sum_HK;
            HL[i * n + j] = sum_HL;
        }
    }

    double hsic = 0.0;
    #pragma omp parallel for reduction(+:hsic)
    for (int i = 0; i < n * n; i++) {
        hsic += HK[i] * HL[i];
    }
    hsic /= (n * n);

    return sqrt(std::max(hsic, 0.0));
}

double compute_hsic(py::array_t<double> x, py::array_t<double> y) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    if (x_buf.ndim != 1 || y_buf.ndim != 1 || x_buf.shape[0] != y_buf.shape[0]) {
        throw std::runtime_error("Input arrays must be 1D and of equal length");
    }
    int n = x_buf.shape[0];
    if (n < 2) return 0.0;

    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);

    try {
        initialize_opencl();

        // Create buffers
        cl::Buffer d_x(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(double), x_ptr);
        cl::Buffer d_y(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(double), y_ptr);
        cl::Buffer d_dist_x(context, CL_MEM_READ_WRITE, n * n * sizeof(double));
        cl::Buffer d_dist_y(context, CL_MEM_READ_WRITE, n * n * sizeof(double));
        cl::Buffer d_K(context, CL_MEM_READ_WRITE, n * n * sizeof(double));
        cl::Buffer d_L(context, CL_MEM_READ_WRITE, n * n * sizeof(double));
        cl::Buffer d_H(context, CL_MEM_READ_WRITE, n * n * sizeof(double));
        cl::Buffer d_HK(context, CL_MEM_READ_WRITE, n * n * sizeof(double));
        cl::Buffer d_HL(context, CL_MEM_READ_WRITE, n * n * sizeof(double));
        cl::Buffer d_temp(context, CL_MEM_READ_WRITE, n * sizeof(double));

        // Compute distance matrices
        cl::Kernel distance_kernel(program, "compute_distance_matrix");
        distance_kernel.setArg(0, d_x);
        distance_kernel.setArg(1, d_dist_x);
        distance_kernel.setArg(2, n);
        cl_int err = queue.enqueueNDRangeKernel(distance_kernel, cl::NullRange, cl::NDRange(n, n), cl::NullRange);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue distance kernel for x: " + std::to_string(err));
        }

        distance_kernel.setArg(0, d_y);
        distance_kernel.setArg(1, d_dist_y);
        err = queue.enqueueNDRangeKernel(distance_kernel, cl::NullRange, cl::NDRange(n, n), cl::NullRange);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue distance kernel for y: " + std::to_string(err));
        }
        queue.finish();

        // Compute sigma (median of non-zero distances)
        std::vector<double> dist_x(n * n);
        err = queue.enqueueReadBuffer(d_dist_x, CL_TRUE, 0, n * n * sizeof(double), dist_x.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read dist_x buffer: " + std::to_string(err));
        }
        std::vector<double> non_zero_dist_x;
        for (double val : dist_x) {
            if (val > 0) non_zero_dist_x.push_back(val);
        }
        double sigma_x = 1.0;
        if (!non_zero_dist_x.empty()) {
            std::sort(non_zero_dist_x.begin(), non_zero_dist_x.end());
            sigma_x = non_zero_dist_x[non_zero_dist_x.size() / 2];
        }

        std::vector<double> dist_y(n * n);
        err = queue.enqueueReadBuffer(d_dist_y, CL_TRUE, 0, n * n * sizeof(double), dist_y.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read dist_y buffer: " + std::to_string(err));
        }
        std::vector<double> non_zero_dist_y;
        for (double val : dist_y) {
            if (val > 0) non_zero_dist_y.push_back(val);
        }
        double sigma_y = 1.0;
        if (!non_zero_dist_y.empty()) {
            std::sort(non_zero_dist_y.begin(), non_zero_dist_y.end());
            sigma_y = non_zero_dist_y[non_zero_dist_y.size() / 2];
        }

        // Compute kernel matrices
        cl::Kernel kernel_matrix(program, "gaussian_kernel_matrix");
        kernel_matrix.setArg(0, d_dist_x);
        kernel_matrix.setArg(1, sigma_x);
        kernel_matrix.setArg(2, d_K);
        kernel_matrix.setArg(3, n);
        err = queue.enqueueNDRangeKernel(kernel_matrix, cl::NullRange, cl::NDRange(n, n), cl::NullRange);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue kernel matrix for K: " + std::to_string(err));
        }

        kernel_matrix.setArg(0, d_dist_y);
        kernel_matrix.setArg(1, sigma_y);
        kernel_matrix.setArg(2, d_L);
        err = queue.enqueueNDRangeKernel(kernel_matrix, cl::NullRange, cl::NDRange(n, n), cl::NullRange);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue kernel matrix for L: " + std::to_string(err));
        }
        queue.finish();

        // Compute centering matrix H
        std::vector<double> H(n * n);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                H[i * n + j] = (i == j) ? 1.0 - 1.0 / n : -1.0 / n;
            }
        }
        err = queue.enqueueWriteBuffer(d_H, CL_TRUE, 0, n * n * sizeof(double), H.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to write H buffer: " + std::to_string(err));
        }

        // Compute HK and HL (matrix multiplication on CPU for simplicity)
        std::vector<double> K(n * n), L(n * n);
        err = queue.enqueueReadBuffer(d_K, CL_TRUE, 0, n * n * sizeof(double), K.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read K buffer: " + std::to_string(err));
        }
        err = queue.enqueueReadBuffer(d_L, CL_TRUE, 0, n * n * sizeof(double), L.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read L buffer: " + std::to_string(err));
        }

        std::vector<double> HK(n * n), HL(n * n);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum_HK = 0.0, sum_HL = 0.0;
                for (int k = 0; k < n; k++) {
                    sum_HK += H[i * n + k] * K[k * n + j];
                    sum_HL += H[i * n + k] * L[k * n + j];
                }
                HK[i * n + j] = sum_HK;
                HL[i * n + j] = sum_HL;
            }
        }

        err = queue.enqueueWriteBuffer(d_HK, CL_TRUE, 0, n * n * sizeof(double), HK.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to write HK buffer: " + std::to_string(err));
        }
        err = queue.enqueueWriteBuffer(d_HL, CL_TRUE, 0, n * n * sizeof(double), HL.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to write HL buffer: " + std::to_string(err));
        }

        // Compute HSIC
        double hsic = 0.0;
        cl::Kernel sum_kernel(program, "reduction_sum");
        sum_kernel.setArg(0, d_HK);
        sum_kernel.setArg(1, d_temp);
        sum_kernel.setArg(2, n * n);
        sum_kernel.setArg(3, sizeof(double) * 256, nullptr);
        err = queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(n * n), cl::NDRange(256));
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue sum kernel for HSIC: " + std::to_string(err));
        }
        queue.finish();

        std::vector<double> temp(n);
        err = queue.enqueueReadBuffer(d_temp, CL_TRUE, 0, n * sizeof(double), temp.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read temp buffer for HSIC: " + std::to_string(err));
        }
        for (double val : temp) hsic += val;
        hsic /= (n * n);

        return sqrt(std::max(hsic, 0.0));

    } catch (const std::exception& e) {
        std::cerr << "OpenCL failed, falling back to CPU: " << e.what() << std::endl;
        return compute_hsic_cpu(x, y);
    }
}

double compute_energy_distance_correlation_cpu(py::array_t<double> x, py::array_t<double> y) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    if (x_buf.ndim != 1 || y_buf.ndim != 1 || x_buf.shape[0] != y_buf.shape[0]) {
        throw std::runtime_error("Input arrays must be 1D and of equal length");
    }
    int n = x_buf.shape[0];
    if (n < 2) return 0.0;

    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);

    std::vector<double> diff_xx(n * n), diff_yy(n * n), diff_xy(n * n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            diff_xx[i * n + j] = fabs(x_ptr[i] - x_ptr[j]);
            diff_yy[i * n + j] = fabs(y_ptr[i] - y_ptr[j]);
            diff_xy[i * n + j] = fabs(x_ptr[i] - y_ptr[j]);
        }
    }

    double dist_xx = 0.0, dist_yy = 0.0, dist_xy = 0.0;
    #pragma omp parallel for reduction(+:dist_xx,dist_yy,dist_xy)
    for (int i = 0; i < n * n; i++) {
        dist_xx += diff_xx[i];
        dist_yy += diff_yy[i];
        dist_xy += diff_xy[i];
    }
    dist_xx /= (n * n);
    dist_yy /= (n * n);
    dist_xy /= (n * n);

    double energy_dist = 2 * dist_xy - dist_xx - dist_yy;

    double var_xx = 0.0, var_yy = 0.0;
    #pragma omp parallel for reduction(+:var_xx,var_yy)
    for (int i = 0; i < n * n; i++) {
        var_xx += (diff_xx[i] - dist_xx) * (diff_xx[i] - dist_xx);
        var_yy += (diff_yy[i] - dist_yy) * (diff_yy[i] - dist_yy);
    }
    var_xx = sqrt(var_xx / (n * n));
    var_yy = sqrt(var_yy / (n * n));

    double energy_corr = (var_xx > 0 && var_yy > 0) ? energy_dist / sqrt(var_xx * var_yy) : 0.0;
    return fabs(energy_corr);
}

double compute_energy_distance_correlation(py::array_t<double> x, py::array_t<double> y) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    if (x_buf.ndim != 1 || y_buf.ndim != 1 || x_buf.shape[0] != y_buf.shape[0]) {
        throw std::runtime_error("Input arrays must be 1D and of equal length");
    }
    int n = x_buf.shape[0];
    if (n < 2) return 0.0;

    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);

    try {
        initialize_opencl();

        // Create buffers
        cl::Buffer d_x(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(double), x_ptr);
        cl::Buffer d_y(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(double), y_ptr);
        cl::Buffer d_diff_xx(context, CL_MEM_READ_WRITE, n * n * sizeof(double));
        cl::Buffer d_diff_yy(context, CL_MEM_READ_WRITE, n * n * sizeof(double));
        cl::Buffer d_temp(context, CL_MEM_READ_WRITE, n * sizeof(double));

        // Compute distance matrices
        cl::Kernel distance_kernel(program, "compute_distance_matrix");
        distance_kernel.setArg(0, d_x);
        distance_kernel.setArg(1, d_diff_xx);
        distance_kernel.setArg(2, n);
        cl_int err = queue.enqueueNDRangeKernel(distance_kernel, cl::NullRange, cl::NDRange(n, n), cl::NullRange);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue distance kernel for xx: " + std::to_string(err));
        }

        distance_kernel.setArg(0, d_y);
        distance_kernel.setArg(1, d_diff_yy);
        err = queue.enqueueNDRangeKernel(distance_kernel, cl::NullRange, cl::NDRange(n, n), cl::NullRange);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue distance kernel for yy: " + std::to_string(err));
        }
        queue.finish();

        // Compute means
        double dist_xx = 0.0, dist_yy = 0.0;
        cl::Kernel sum_kernel(program, "reduction_sum");
        sum_kernel.setArg(0, d_diff_xx);
        sum_kernel.setArg(1, d_temp);
        sum_kernel.setArg(2, n * n);
        sum_kernel.setArg(3, sizeof(double) * 256, nullptr);
        err = queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(n * n), cl::NDRange(256));
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue sum kernel for xx: " + std::to_string(err));
        }
        queue.finish();

        std::vector<double> temp(n);
        err = queue.enqueueReadBuffer(d_temp, CL_TRUE, 0, n * sizeof(double), temp.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read temp buffer for xx: " + std::to_string(err));
        }
        for (double val : temp) dist_xx += val;
        dist_xx /= (n * n);

        sum_kernel.setArg(0, d_diff_yy);
        err = queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(n * n), cl::NDRange(256));
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue sum kernel for yy: " + std::to_string(err));
        }
        queue.finish();
        err = queue.enqueueReadBuffer(d_temp, CL_TRUE, 0, n * sizeof(double), temp.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read temp buffer for yy: " + std::to_string(err));
        }
        for (double val : temp) dist_yy += val;
        dist_yy /= (n * n);

        // Compute dist_xy
        double dist_xy = 0.0;
        distance_kernel.setArg(0, d_x);
        distance_kernel.setArg(1, d_diff_xx); // Reuse buffer
        err = queue.enqueueNDRangeKernel(distance_kernel, cl::NullRange, cl::NDRange(n, n), cl::NullRange);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue distance kernel for xy: " + std::to_string(err));
        }
        queue.finish();
        sum_kernel.setArg(0, d_diff_xx);
        err = queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(n * n), cl::NDRange(256));
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue sum kernel for xy: " + std::to_string(err));
        }
        queue.finish();
        err = queue.enqueueReadBuffer(d_temp, CL_TRUE, 0, n * sizeof(double), temp.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read temp buffer for xy: " + std::to_string(err));
        }
        for (double val : temp) dist_xy += val;
        dist_xy /= (n * n);

        double energy_dist = 2 * dist_xy - dist_xx - dist_yy;

        // Compute variances
        double var_xx = 0.0, var_yy = 0.0;
        sum_kernel.setArg(0, d_diff_xx);
        err = queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(n * n), cl::NDRange(256));
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue sum kernel for var_xx: " + std::to_string(err));
        }
        queue.finish();
        err = queue.enqueueReadBuffer(d_temp, CL_TRUE, 0, n * sizeof(double), temp.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read temp buffer for var_xx: " + std::to_string(err));
        }
        for (double val : temp) var_xx += (val - dist_xx) * (val - dist_xx);
        var_xx = sqrt(var_xx / (n * n));

        sum_kernel.setArg(0, d_diff_yy);
        err = queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(n * n), cl::NDRange(256));
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue sum kernel for var_yy: " + std::to_string(err));
        }
        queue.finish();
        err = queue.enqueueReadBuffer(d_temp, CL_TRUE, 0, n * sizeof(double), temp.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read temp buffer for var_yy: " + std::to_string(err));
        }
        for (double val : temp) var_yy += (val - dist_yy) * (val - dist_yy);
        var_yy = sqrt(var_yy / (n * n));

        double energy_corr = (var_xx > 0 && var_yy > 0) ? energy_dist / sqrt(var_xx * var_yy) : 0.0;
        return fabs(energy_corr);

    } catch (const std::exception& e) {
        std::cerr << "OpenCL failed, falling back to CPU: " << e.what() << std::endl;
        return compute_energy_distance_correlation_cpu(x, y);
    }
}

PYBIND11_MODULE(metrics, m) {
    m.def("compute_distance_correlation", &compute_distance_correlation, 
          "Compute distance correlation between two arrays");
    m.def("compute_mic", &compute_mic, 
          "Compute Maximal Information Coefficient between two arrays", 
          py::arg("x"), py::arg("y"), py::arg("max_bins")=10);
    m.def("compute_hsic", &compute_hsic, 
          "Compute Hilbert-Schmidt Independence Criterion between two arrays");
    m.def("compute_copula_measure", &compute_copula_measure, 
          "Compute copula-based dependence measure between two arrays");
    m.def("compute_energy_distance_correlation", &compute_energy_distance_correlation, 
          "Compute energy distance correlation between two arrays");
}
