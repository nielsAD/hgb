// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "alg/pagerank/problem.h"
#include "util/cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_pr_problem {
    pr_bcsr_graph_t *graph;

    cudaStream_t streams[BCSR_GRAPH_MAX_BCOUNT];

    pr_float *data_global_f[E_PR_PROBLEM_GLOBAL_MAX];
    pr_float *data_blocks_f[E_PR_PROBLEM_BLOCKS_MAX][BCSR_GRAPH_MAX_BCOUNT*BCSR_GRAPH_MAX_BCOUNT];
    graph_size_t *data_global_u[E_PR_PROBLEM_GLOBAL_MAX];
    graph_size_t *data_blocks_u[E_PR_PROBLEM_BLOCKS_MAX][BCSR_GRAPH_MAX_BCOUNT*BCSR_GRAPH_MAX_BCOUNT];
} cuda_pr_problem_t;

cuda_pr_problem_t *cuda_pr_problem_new(void);
cuda_pr_problem_t *cuda_pr_problem_new_bcsc(const pr_bcsc_graph_t *graph);
cuda_pr_problem_t *cuda_pr_problem_new_bcsr(const pr_bcsr_graph_t *graph);
void cuda_pr_problem_free(cuda_pr_problem_t *problem);

void cuda_pr_problem_synchronize(const cuda_pr_problem_t *problem);

#ifdef __cplusplus
}
#endif