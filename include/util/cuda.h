// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "util/starpu.h"
#define CUDA_CHECK_ERROR STARPU_CUDA_CHECK_ERROR

#ifdef NDEBUG
	#define CUDA_ASSERT(x) { if(x){} }
#else
	#define CUDA_ASSERT(x) CUDA_CHECK_ERROR((cudaError_t)(x))
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern cublasHandle_t   handle_cublas;
extern cusparseHandle_t handle_cusparse;

void cuda_initialize(void);
void cuda_finalize(void);

#ifdef __cplusplus
}
#endif