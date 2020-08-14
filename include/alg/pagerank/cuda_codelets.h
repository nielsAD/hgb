// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "alg/pagerank/codelets.h"
#include "util/cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_pagerank_read_col(const cudaStream_t stream, pr_float *dst, const pr_float *src, const graph_size_t dst_offset, const graph_size_t dst_cols, const graph_size_t src_cols, const graph_size_t size);
void cuda_pagerank_fill_arr(const cudaStream_t stream, pr_float *arr, const pr_float val, const graph_size_t size);
void cuda_pagerank_add_arr(const cudaStream_t stream, pr_float *restrict a, const pr_float *restrict b, const graph_size_t size);
void cuda_pagerank_sum_arr(const cudaStream_t stream, pr_float *res, const pr_float *arr, const graph_size_t size);
void cuda_pagerank_sum_arr_offset(const cudaStream_t stream, pr_float *res, const pr_float *arr, const graph_size_t offset, const graph_size_t size);
void cuda_pagerank_parallel_sum_arr(const cudaStream_t stream, pr_float *res, const pr_float *arr, pr_float *scr, const graph_size_t arr_size, const graph_size_t scr_size);
void cuda_pagerank_parallel_sum_arr_offset(const cudaStream_t stream, pr_float *res, const pr_float *arr, pr_float *scr, const graph_size_t offset, const graph_size_t arr_size, const graph_size_t scr_size);
void cuda_pagerank_baserank(const cudaStream_t stream, pr_float *restrict res, const pr_float *restrict src, const graph_size_t *restrict deg, const graph_size_t vcount);
void cuda_pagerank_baserank_redux(const cudaStream_t stream, pr_float *redux, const pr_float damping, const graph_size_t vcount);
void cuda_pagerank_fill_cols_pull(const cudaStream_t stream, pr_float *restrict dst, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t *restrict deg, const graph_size_t vcount);
void cuda_pagerank_fill_cols_push(const cudaStream_t stream, pr_float *restrict dst, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t *restrict deg, const graph_size_t vcount);
void cuda_pagerank_update_rank_pull(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict src, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t *restrict deg, const graph_size_t vcount);
void cuda_pagerank_update_rank_push(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict src, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t *restrict deg, const graph_size_t vcount);
void cuda_pagerank_update_tmp(const cudaStream_t stream, pr_float *restrict tmp, const pr_float *restrict src, const graph_size_t *restrict deg, const graph_size_t vcount);
void cuda_pagerank_update_rank_tmp_pull(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount);
void cuda_pagerank_update_rank_tmp_push(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount);
void cuda_pagerank_update_rank_tmp_pull_warp(const cudaStream_t stream, const graph_size_t avg, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount);
void cuda_pagerank_update_rank_tmp_push_warp(const cudaStream_t stream, const graph_size_t avg, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount);
void cuda_pagerank_update_rank_tmp_pull_dyn(const cudaStream_t stream, const graph_size_t avg, graph_size_t *restrict scr, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount);
void cuda_pagerank_update_rank_tmp_push_dyn(const cudaStream_t stream, const graph_size_t avg, graph_size_t *restrict scr, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount);
void cuda_pagerank_update_dest(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict brp, const pr_float damping, const graph_size_t vcount);
void cuda_pagerank_update_dest_raw(const cudaStream_t stream, pr_float *restrict dst, const pr_float brp, const pr_float damping, const graph_size_t vcount);
void cuda_pagerank_calc_dest(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict tmp, const pr_float *restrict brp, const pr_float damping, const graph_size_t vcount);
void cuda_pagerank_calc_dest_raw(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict tmp, const pr_float brp, const pr_float damping, const graph_size_t vcount);
void cuda_pagerank_calc_diff(const cudaStream_t stream, pr_float *restrict dif, const pr_float *restrict src, const pr_float *restrict dst, const graph_size_t vcount);

#ifdef __cplusplus
}
#endif