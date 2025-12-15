cdef extern from "inequalities.h":
    void compute_features(double* data, int rows, int* cols, int num_cols, int level, 
                         int stage, double* output, int* output_cols, char** output_names)
    const int num_inequalities

cdef extern from "stdlib.h":
    void* malloc(size_t size)
    void free(void* ptr)

cdef extern from "string.h":
    char* strdup(const char* s)

import numpy as np
cimport numpy as np
cimport cython
np.import_array()
from libc.stdlib cimport malloc, free
from libc.string cimport strdup

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_features_cython(np.ndarray[np.double_t, ndim=2] data, 
                          np.ndarray[np.int32_t, ndim=1] cols, 
                          int level, int stage):
    cdef int rows = data.shape[0]
    cdef int num_cols = cols.shape[0]
    cdef int max_output_cols = num_inequalities * (num_cols * (num_cols + 1)) // 2
    cdef np.ndarray[np.double_t, ndim=2] output = np.zeros((rows, max_output_cols), dtype=np.double)
    cdef int output_cols = 0
    cdef char** output_names = <char**>malloc(max_output_cols * sizeof(char*))
    if output_names == NULL:
        raise MemoryError("Failed to allocate memory for output_names")
    
    # Convert cols to int* for C compatibility
    cdef int* cols_c = <int*>malloc(num_cols * sizeof(int))
    if cols_c == NULL:
        free(output_names)
        raise MemoryError("Failed to allocate memory for cols_c")
    for i in range(num_cols):
        cols_c[i] = cols[i]
    
    try:
        compute_features(&data[0, 0], rows, cols_c, num_cols, level, stage, 
                        &output[0, 0], &output_cols, output_names)
        
        names = []
        for i in range(output_cols):
            names.append(<bytes>output_names[i].decode('utf-8'))
            free(output_names[i])
        
        return output[:, :output_cols], names
    finally:
        free(cols_c)
        free(output_names)
