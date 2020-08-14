// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "alg/pagerank/cpu_codelets.h"

#define OMP_PAGERANK_TIME_DECLARE() \
    time_diff_t __omp_max[E_PR_STAGE_MAX] = {0}; \
    size_t      __omp_cnt[E_PR_STAGE_MAX] = {0};

#define OMP_PAGERANK_TIME_START(x) \
    PAGERANK_TIME_START(x) \
    OMP_PRAGMA(omp atomic) \
    __omp_cnt[E_PR_STAGE_ ## x]++;

#define OMP_PAGERANK_TIME_STOP(x) { \
    time_diff_t d = PAGERANK_TIME_SINCE(x); \
    OMP_PRAGMA(omp critical) \
    { \
        if (d > __omp_max[E_PR_STAGE_ ## x]) __omp_max[E_PR_STAGE_ ## x] = d; \
        if (--__omp_cnt[E_PR_STAGE_ ## x] == 0) { \
            options->stage_time[E_PR_STAGE_ ## x] += __omp_max[E_PR_STAGE_ ## x]; \
            __omp_max[E_PR_STAGE_ ## x] = 0; \
        } \
    }} \

#ifdef __cplusplus
extern "C" {
#endif

void omp_pagerank_read_col(pr_float *restrict _dst, const pr_float *restrict _src, const graph_size_t dst_offset, const graph_size_t dst_cols, const graph_size_t src_cols, const graph_size_t size);
void omp_pagerank_fill_arr(pr_float *_arr, const pr_float val, const graph_size_t size);
void omp_pagerank_add_arr(pr_float *_a, const pr_float *_b, const graph_size_t size);
pr_float omp_pagerank_sum_arr(const pr_float *_arr, const graph_size_t size);
pr_float omp_pagerank_baserank(const pr_float *_src, const graph_size_t *_deg, const graph_size_t vcount);
void omp_pagerank_baserank_mapped(pr_float *_tmp, const pr_float *_src, const graph_size_t *_deg, const graph_size_t vcount);
void omp_pagerank_update_rank_pull(pr_float *_dst, const pr_float *_src, const graph_size_t *_rid, const graph_size_t *_cid, const graph_size_t *_deg, const graph_size_t vcount);
void omp_pagerank_update_rank_push(pr_float *_dst, const pr_float *_src, const graph_size_t *_rid, const graph_size_t *_cid, const graph_size_t *_deg, const graph_size_t vcount);
void omp_pagerank_update_tmp(pr_float *_tmp, const pr_float *_src, const graph_size_t *_deg, const graph_size_t vcount);
void omp_pagerank_update_rank_tmp_pull(pr_float *_dst, const pr_float *_tmp, const graph_size_t *_rid, const graph_size_t *_cid, const graph_size_t vcount);
void omp_pagerank_update_rank_tmp_push(pr_float *_dst, const pr_float *_tmp, const graph_size_t *_rid, const graph_size_t *_cid, const graph_size_t vcount);
void omp_pagerank_update_rank_tmp_pull_binsearch(pr_float *_dst, const pr_float *_tmp, const graph_size_t *_rid, const graph_size_t *_cid, const graph_size_t vcount);
void omp_pagerank_update_rank_tmp_push_binsearch(pr_float *_dst, const pr_float *_tmp, const graph_size_t *_rid, const graph_size_t *_cid, const graph_size_t vcount);
void omp_pagerank_update_dest(pr_float *_dst, const pr_float base_rank, const pr_float damping, const graph_size_t vcount);
void omp_pagerank_calc_dest(pr_float *_dst, pr_float *_tmp, const pr_float base_rank, const pr_float damping, const graph_size_t vcount);
pr_float omp_pagerank_calc_diff(const pr_float *_src, const pr_float *_dst, const graph_size_t vcount);
void omp_pagerank_calc_diff_mapped(pr_float *_tmp, const pr_float *_src, const pr_float *_dst, const graph_size_t vcount);

#ifdef __cplusplus
}
#endif