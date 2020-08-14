// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "alg/pagerank/spu_problem.h"

#ifdef __cplusplus
extern "C" {
#endif

void spu_pagerank_read_col_cpu(void *buffers[], void *args);
void spu_pagerank_read_col_ocl(void *buffers[], void *args);
void spu_pagerank_read_col_cuda(void *buffers[], void *args);

void spu_pagerank_fill_arr_cpu(void *buffers[], void *args);
void spu_pagerank_fill_arr_ocl(void *buffers[], void *args);
void spu_pagerank_fill_arr_cuda(void *buffers[], void *args);

void spu_pagerank_redux_zero_single_cpu(void *buffers[], void *args);
void spu_pagerank_redux_zero_single_ocl(void *buffers[], void *args);
void spu_pagerank_redux_zero_single_cuda(void *buffers[], void *args);

void spu_pagerank_redux_zero_cpu(void *buffers[], void *args);
void spu_pagerank_redux_zero_ocl(void *buffers[], void *args);
void spu_pagerank_redux_zero_cuda(void *buffers[], void *args);

void spu_pagerank_redux_add_single_cpu(void *buffers[], void *args);
void spu_pagerank_redux_add_single_ocl(void *buffers[], void *args);
void spu_pagerank_redux_add_single_cuda(void *buffers[], void *args);

void spu_pagerank_redux_add_cpu(void *buffers[], void *args);
void spu_pagerank_redux_add_ocl(void *buffers[], void *args);
void spu_pagerank_redux_add_cuda(void *buffers[], void *args);

void spu_pagerank_redux_sum_cpu(void *buffers[], void *args);
void spu_pagerank_redux_sum_ocl(void *buffers[], void *args);
void spu_pagerank_redux_sum_cuda(void *buffers[], void *args);
void spu_pagerank_redux_parallel_sum_ocl(void *buffers[], void *args);
void spu_pagerank_redux_parallel_sum_cuda(void *buffers[], void *args);

void spu_pagerank_baserank_cpu(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_baserank_ocl(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_baserank_cuda(void *buffers[], spu_pr_problem_t *problem);

void spu_pagerank_baserank_redux_cpu(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_baserank_redux_ocl(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_baserank_redux_cuda(void *buffers[], spu_pr_problem_t *problem);

void spu_pagerank_update_rank_pull_cpu(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_pull_ocl(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_pull_cuda(void *buffers[], spu_pr_problem_t *problem);

void spu_pagerank_update_rank_push_cpu(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_push_ocl(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_push_cuda(void *buffers[], spu_pr_problem_t *problem);

void spu_pagerank_update_tmp_cpu(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_tmp_ocl(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_tmp_cuda(void *buffers[], spu_pr_problem_t *problem);

void spu_pagerank_update_rank_tmp_pull_cpu(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_tmp_pull_ocl(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_tmp_pull_cuda(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_tmp_pull_warp_ocl(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_tmp_pull_warp_cuda(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_tmp_pull_dyn_cuda(void *buffers[], spu_pr_problem_t *problem);

void spu_pagerank_update_rank_tmp_push_cpu(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_tmp_push_ocl(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_tmp_push_cuda(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_tmp_push_warp_ocl(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_tmp_push_warp_cuda(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_rank_tmp_push_dyn_cuda(void *buffers[], spu_pr_problem_t *problem);

void spu_pagerank_update_dest_cpu(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_dest_ocl(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_update_dest_cuda(void *buffers[], spu_pr_problem_t *problem);

void spu_pagerank_calc_dest_cpu(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_calc_dest_ocl(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_calc_dest_cuda(void *buffers[], spu_pr_problem_t *problem);

void spu_pagerank_calc_diff_cpu(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_calc_diff_ocl(void *buffers[], spu_pr_problem_t *problem);
void spu_pagerank_calc_diff_cuda(void *buffers[], spu_pr_problem_t *problem);

#ifdef __cplusplus
}
#endif