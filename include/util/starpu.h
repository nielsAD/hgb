// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"
#include <starpu.h>

#define STARPU_OPENCL_CHECK_ERROR(ERR){ if (UNLIKELY((ERR) != CL_SUCCESS))  STARPU_OPENCL_REPORT_ERROR((ERR));}
#define STARPU_CUDA_CHECK_ERROR(ERR)  { if (UNLIKELY((ERR) != cudaSuccess)) STARPU_CUDA_REPORT_ERROR((ERR));}

#define STARPU_VECTOR_GET_OFFSET_IDX(D)(STARPU_VECTOR_GET_OFFSET((D)) / STARPU_VECTOR_GET_ELEMSIZE((D)))

#ifdef STARPU_OPENCL_SYNC
    #undef STARPU_OPENCL_ASYNC
    #define STARPU_OPENCL_ASYNC 0
#endif

#ifdef STARPU_CUDA_SYNC
    #undef STARPU_CUDA_ASYNC
    #define STARPU_CUDA_ASYNC 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

void starpu_inititialize_conf(int argc, char *argv[]);
void starpu_finalize(void);

extern size_t _starpu_free_all_automatically_allocated_buffers(unsigned node);

#ifdef __cplusplus
}
#endif