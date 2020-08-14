// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "util/cuda.h"

cublasHandle_t   handle_cublas   = NULL;
cusparseHandle_t handle_cusparse = NULL;

void cuda_initialize(void)
{
    CUDA_ASSERT(cublasCreate(&handle_cublas));
    CUDA_ASSERT(cusparseCreate(&handle_cusparse));
}

void cuda_finalize(void)
{
    if (handle_cusparse) CUDA_ASSERT(cusparseDestroy(handle_cusparse));
    if (handle_cublas)   CUDA_ASSERT(cublasDestroy(handle_cublas));
}
