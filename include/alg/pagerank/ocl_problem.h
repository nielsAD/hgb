// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "alg/pagerank/problem.h"
#include "util/opencl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ocl_pr_problem {
    pr_bcsr_graph_t *graph;

    cl_context context;
    cl_command_queue queue;

    cl_mem data_global[E_PR_PROBLEM_GLOBAL_MAX];
    cl_mem data_blocks[E_PR_PROBLEM_BLOCKS_MAX][BCSR_GRAPH_MAX_BCOUNT*BCSR_GRAPH_MAX_BCOUNT];

    cl_event event_global[E_PR_PROBLEM_GLOBAL_MAX][BCSR_GRAPH_MAX_BCOUNT*BCSR_GRAPH_MAX_BCOUNT];
    cl_event event_blocks[E_PR_PROBLEM_BLOCKS_MAX][BCSR_GRAPH_MAX_BCOUNT*BCSR_GRAPH_MAX_BCOUNT];
} ocl_pr_problem_t;

ocl_pr_problem_t *ocl_pr_problem_new(const int devid);
ocl_pr_problem_t *ocl_pr_problem_new_bcsr(const int devid, const pr_bcsr_graph_t *graph);
ocl_pr_problem_t *ocl_pr_problem_new_bcsc(const int devid, const pr_bcsc_graph_t *graph);
void ocl_pr_problem_free(ocl_pr_problem_t *problem);

void ocl_pr_problem_clear_events(ocl_pr_problem_t *problem);
cl_event ocl_pr_problem_set_event_g(ocl_pr_problem_t *problem, const pagerank_problem_data_global_enum_t index, const graph_size_t block, const cl_event event);
cl_event ocl_pr_problem_set_event_b(ocl_pr_problem_t *problem, const pagerank_problem_data_block_enum_t index, const graph_size_t block, const cl_event event);

#ifdef __cplusplus
}
#endif