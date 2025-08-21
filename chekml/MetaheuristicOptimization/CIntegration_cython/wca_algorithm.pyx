# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt
from time import time

cnp.import_array()

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t ITYPE_t

cdef class WaterCycleAlgorithm:
    cdef object objective_function
    cdef int dim
    cdef double[:, ::1] bounds
    cdef int population_size
    cdef int max_iter
    cdef bint use_lsar_asp
    cdef int max_lsar_iter
    cdef double[:, ::1] streams
    cdef double[::1] best_solution
    cdef double best_value
    cdef list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100,
                 bint use_lsar_asp=True, int max_lsar_iter=2000):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.use_lsar_asp = use_lsar_asp
        self.max_lsar_iter = max_lsar_iter
        self.streams = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_streams(self):
        self.streams = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                        (self.population_size, self.dim)).astype(np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_streams(self):
        cdef cnp.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=np.float64)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.streams[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double iop(self, double[:, ::1] C, double[:, ::1] D):
        if C.size == 0:
            return 0
        cdef cnp.ndarray[DTYPE_t, ndim=1] count_C = self.equivalent_class(C)
        cdef cnp.ndarray[DTYPE_t, ndim=1] count_CD = self.equivalent_class(
            np.hstack([C, D]).astype(np.float64))
        cdef cnp.ndarray[DTYPE_t, ndim=1] count = count_C - count_CD
        cdef double sum_count = 0.0
        cdef int i
        for i in range(count.shape[0]):
            sum_count += count[i]
        return sum_count

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[DTYPE_t, ndim=1] equivalent_class(self, double[:, ::1] C):
        cdef double[:, ::1] C_contig = np.ascontiguousarray(C)
        cdef int n_rows = C_contig.shape[0]
        cdef int n_cols = C_contig.shape[1]
        cdef cnp.ndarray structured = np.zeros(n_rows, dtype=[('f{}'.format(i), np.float64) for i in range(n_cols)])
        cdef int i, j
        for i in range(n_rows):
            for j in range(n_cols):
                structured['f{}'.format(j)][i] = C_contig[i, j]
        cdef cnp.ndarray[ITYPE_t, ndim=1] inv
        cdef cnp.ndarray[ITYPE_t, ndim=1] counts
        _, inv, counts = np.unique(structured, return_inverse=True, return_counts=True)
        return counts[inv].astype(np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[ITYPE_t, ndim=1] positive_region(self, double[:, ::1] C, double[:, ::1] D):
        if C.size == 0:
            return np.array([], dtype=np.int64)
        cdef cnp.ndarray[DTYPE_t, ndim=1] count_C = self.equivalent_class(C)
        cdef cnp.ndarray[DTYPE_t, ndim=1] count_CD = self.equivalent_class(
            np.hstack([C, D]).astype(np.float64))
        cdef cnp.ndarray[DTYPE_t, ndim=1] count = count_C - count_CD
        return np.where(count == 0)[0].astype(np.int64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long random_select_att(self, long[::1] B):
        if B.shape[0] == 0:
            raise ValueError("Error: Empty input set")
        return B[rand() % B.shape[0]]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[ITYPE_t, ndim=1] fast_red(self, double[:, ::1] C, double[:, ::1] D):
        cdef int att = C.shape[1]
        cdef double iop_C = self.iop(C, D)
        cdef cnp.ndarray[DTYPE_t, ndim=1] w = np.zeros(att, dtype=np.float64)
        cdef int i
        for i in range(att):
            w[i] = self.iop(C[:, i:i+1], D)
        cdef cnp.ndarray[ITYPE_t, ndim=1] ind = np.argsort(w).astype(np.int64)
        cdef cnp.ndarray[ITYPE_t, ndim=1] red = np.array([], dtype=np.int64)
        for i in range(att):
            red = np.union1d(red, [ind[i]]).astype(np.int64)
            C_red = np.ascontiguousarray(np.take(C, red, axis=1))
            if self.iop(C_red, D) == iop_C:
                break
        return red

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef (long, long) aps_mechanism(self, double[:, ::1] C, double[:, ::1] D, long[::1] B):
        cdef long u = 0, v = 0
        cdef int att = C.shape[1]
        cdef cnp.ndarray[ITYPE_t, ndim=1] pos = self.positive_region(C, D)
        cdef cnp.ndarray[ITYPE_t, ndim=1] B_np = np.array(B, dtype=np.int64)
        C_B = np.ascontiguousarray(np.take(C, B_np, axis=1))
        cdef cnp.ndarray[ITYPE_t, ndim=1] unpos = np.setdiff1d(pos, self.positive_region(C_B, D)).astype(np.int64)
        cdef cnp.ndarray[ITYPE_t, ndim=1] unred = np.setdiff1d(np.arange(att), B_np).astype(np.int64)
        cdef cnp.ndarray[ITYPE_t, ndim=1] add = np.array([], dtype=np.int64)
        cdef long k, i, j
        for k in unred:
            C_unpos = np.ascontiguousarray(np.take(C, unpos, axis=0))
            D_unpos = np.ascontiguousarray(np.take(D, unpos, axis=0))
            C_unpos_k = np.ascontiguousarray(C_unpos[:, [k]])
            if len(self.positive_region(C_unpos_k, D_unpos)) == unpos.shape[0]:
                add = np.union1d(add, [k]).astype(np.int64)
        for i in add:
            subspace = np.union1d(B_np, [i]).astype(np.int64)
            unique_rows = np.unique(np.take(C, subspace, axis=1), axis=0, return_index=True)[1]
            U = np.sort(unique_rows).astype(np.int64)
            for j in B_np:
                testB = np.setdiff1d(np.union1d(B_np, [i]), [j]).astype(np.int64)
                C_U = np.ascontiguousarray(np.take(C, U, axis=0))
                D_U = np.ascontiguousarray(np.take(D, U, axis=0))
                C_U_testB = np.ascontiguousarray(np.take(C_U, testB, axis=1))
                if len(self.positive_region(C_U_testB, D_U)) == U.shape[0]:
                    v = i
                    u = j
                    break
            if u != 0 or v != 0:
                break
        return u, v

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[ITYPE_t, ndim=1] lsar(self, double[:, ::1] C, double[:, ::1] D, int T):
        cdef double start_time = time()
        cdef int att = C.shape[1]
        cdef cnp.ndarray[ITYPE_t, ndim=1] red = self.fast_red(C, D)
        cdef cnp.ndarray[ITYPE_t, ndim=1] pos = self.positive_region(C, D)
        cdef int t = 0
        cdef long a, u, v
        while t < T:
            a = self.random_select_att(red)
            diff_red = np.setdiff1d(red, a).astype(np.int64)
            C_diff_red = np.ascontiguousarray(np.take(C, diff_red, axis=1))
            if len(self.positive_region(C_diff_red, D)) == pos.shape[0]:
                red = diff_red
            else:
                diff_red_np = np.array(diff_red, dtype=np.int64)
                u = self.random_select_att(diff_red_np)
                v = self.random_select_att(np.setdiff1d(np.arange(att), diff_red_np).astype(np.int64))
                new_red = np.union1d(np.setdiff1d(diff_red_np, u), v).astype(np.int64)
                C_new_red = np.ascontiguousarray(np.take(C, new_red, axis=1))
                if len(self.positive_region(C_new_red, D)) == pos.shape[0]:
                    red = new_red
            t += 1
        return red

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[ITYPE_t, ndim=1] lsar_asp(self, double[:, ::1] C, double[:, ::1] D):
        cdef double start_time = time()
        cdef cnp.ndarray[ITYPE_t, ndim=1] RemoveSet = np.array([], dtype=np.int64)
        cdef cnp.ndarray[ITYPE_t, ndim=1] red = self.fast_red(C, D)
        cdef cnp.ndarray[ITYPE_t, ndim=1] pos = self.positive_region(C, D)
        cdef int t = 0
        while len(red) != len(RemoveSet):
            diff_red = np.setdiff1d(red, RemoveSet).astype(np.int64)
            a = self.random_select_att(diff_red)
            diff_red_a = np.setdiff1d(red, a).astype(np.int64)
            C_diff_red_a = np.ascontiguousarray(np.take(C, diff_red_a, axis=1))
            if len(self.positive_region(C_diff_red_a, D)) == pos.shape[0]:
                red = diff_red_a
            else:
                u, v = self.aps_mechanism(C, D, diff_red_a)
                if u == 0 and v == 0:
                    RemoveSet = np.union1d(RemoveSet, [a]).astype(np.int64)
                else:
                    red = np.union1d(np.setdiff1d(np.setdiff1d(red, a), u), v).astype(np.int64)
                    RemoveSet = np.array([], dtype=np.int64)
            t += 1
        return red

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        self.initialize_streams()
        cdef int generation, i, j, min_idx
        cdef double[:, ::1] C
        cdef double[:, ::1] D
        cdef cnp.ndarray[DTYPE_t, ndim=1] fitness
        cdef cnp.ndarray[ITYPE_t, ndim=1] red
        for generation in range(self.max_iter):
            fitness = self.evaluate_streams()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.streams[min_idx].copy()
                self.best_value = fitness[min_idx]

            C = self.streams
            D = fitness.reshape(-1, 1)
            if self.use_lsar_asp:
                red = self.lsar_asp(C, D)
            else:
                red = self.lsar(C, D, self.max_lsar_iter)

            for i in range(self.population_size):
                if np.random.rand() < 0.5:
                    for j in red:
                        self.streams[i, j] = self.best_solution[j] + \
                            np.random.uniform(-1, 1) * (self.bounds[j, 1] - self.bounds[j, 0]) * 0.1
                for j in range(self.dim):
                    if self.streams[i, j] < self.bounds[j, 0]:
                        self.streams[i, j] = self.bounds[j, 0]
                    elif self.streams[i, j] > self.bounds[j, 1]:
                        self.streams[i, j] = self.bounds[j, 1]

            self.history.append((generation, np.array(self.best_solution), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return np.array(self.best_solution), self.best_value, self.history
