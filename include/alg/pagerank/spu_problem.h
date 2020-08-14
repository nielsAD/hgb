// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "alg/pagerank/problem.h"
#include "util/starpu.h"

#include <semaphore.h>

#ifdef __cplusplus
extern "C" {
#endif

struct spu_pr_problem;

typedef struct spu_pr_problem {
    pr_bcsr_graph_t *graph;
    void(*iteration_func)(struct spu_pr_problem *problem);

    pagerank_options_t *options;

    bool     pull;
    uint32_t iterations;
    pr_float diff_sum;

    starpu_data_handle_t data_global[E_PR_PROBLEM_GLOBAL_MAX];
    starpu_data_handle_t data_blocks[E_PR_PROBLEM_BLOCKS_MAX][BCSR_GRAPH_MAX_BCOUNT*BCSR_GRAPH_MAX_BCOUNT];

    sem_t semaphore;
} spu_pr_problem_t;

typedef void(*spu_pr_problem_iteration_func_t)(spu_pr_problem_t *problem);

spu_pr_problem_t *spu_pr_problem_new(const spu_pr_problem_iteration_func_t iteration_func, const bool pull, pagerank_options_t *options);
spu_pr_problem_t *spu_pr_problem_new_bcsc(const pr_bcsc_graph_t *graph, const spu_pr_problem_iteration_func_t iteration_func, pagerank_options_t *options);
spu_pr_problem_t *spu_pr_problem_new_bcsr(const pr_bcsr_graph_t *graph, const spu_pr_problem_iteration_func_t iteration_func, pagerank_options_t *options);
void spu_pr_problem_free(spu_pr_problem_t *problem);
void spu_pr_problem_iteration(spu_pr_problem_t *problem);
void spu_pr_problem_callback(void *problem);

#ifdef __cplusplus
}
#endif