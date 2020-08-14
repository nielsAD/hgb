// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"

#ifdef _OPENMP
    #include <omp.h>
    #define OMP_PRAGMA(...) _Pragma(#__VA_ARGS__)
#else
    #include <time.h>

    #warning OpenMP not supported on this platform
    #define OMP_PRAGMA(...)

    static inline int omp_get_num_threads(void) { return 1; }
    static inline int omp_get_max_threads(void) { return 1; }
    static inline int omp_get_thread_num(void)  { return 0; }

    static inline int omp_get_dynamic(void) { return 0; }
    static inline void omp_set_dynamic(UNUSED int v) { /* ignore */ }

    static inline double omp_get_wtime(void) {
        return (double) time(NULL);
    }
#endif

static const unsigned int OMP_TASK_CUTOFF = 1024;
