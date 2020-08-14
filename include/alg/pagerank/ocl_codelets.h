// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "alg/pagerank/codelets.h"
#include "util/opencl.h"
#include "util/starpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ocl_pagerank_kernel_index {
    E_PR_PROBLEM_KERNEL_COPY_COL = 0,
    E_PR_PROBLEM_KERNEL_FILL_ARR,
    E_PR_PROBLEM_KERNEL_ADD_ARR,
    E_PR_PROBLEM_KERNEL_SUM_ARR,
    E_PR_PROBLEM_KERNEL_SUM_ARR_OFFSET,
    E_PR_PROBLEM_KERNEL_SUM_ARR_PARALLEL,
    E_PR_PROBLEM_KERNEL_SUM_ARR_PARALLEL_OFFSET,
    E_PR_PROBLEM_KERNEL_BASERANK,
    E_PR_PROBLEM_KERNEL_BASERANK_REDUX,
    E_PR_PROBLEM_KERNEL_UPDATE_RANK_PULL,
    E_PR_PROBLEM_KERNEL_UPDATE_RANK_PUSH,
    E_PR_PROBLEM_KERNEL_UPDATE_TMP,
    E_PR_PROBLEM_KERNEL_UPDATE_RANK_TMP_PULL,
    E_PR_PROBLEM_KERNEL_UPDATE_RANK_TMP_PUSH,
    E_PR_PROBLEM_KERNEL_UPDATE_RANK_TMP_PULL_WARP,
    E_PR_PROBLEM_KERNEL_UPDATE_RANK_TMP_PUSH_WARP,
    E_PR_PROBLEM_KERNEL_UPDATE_DEST,
    E_PR_PROBLEM_KERNEL_UPDATE_DEST_RAW,
    E_PR_PROBLEM_KERNEL_CALC_DEST,
    E_PR_PROBLEM_KERNEL_CALC_DEST_RAW,
    E_PR_PROBLEM_KERNEL_CALC_DIFF,
    E_PR_PROBLEM_KERNEL_MAX
} ocl_pagerank_kernel_index_enum_t;

static const char *const ocl_pagerank_kernel_names[] = {
    NULL,
    "pagerank_fill_arr_single",
    "pagerank_redux_add_single",
    "pagerank_redux_sum",
    "pagerank_redux_sum_offset",
    "pagerank_redux_sum_parallel",
    NULL,
    "pagerank_baserank_single",
    "pagerank_baserank_redux",
    "pagerank_update_rank_pull_single",
    "pagerank_update_rank_push_single",
    "pagerank_update_tmp_single",
    "pagerank_update_rank_tmp_pull_single",
    "pagerank_update_rank_tmp_push_single",
    "pagerank_update_rank_tmp_pull_warp",
    "pagerank_update_rank_tmp_push_warp",
    "pagerank_update_dest_single",
    "pagerank_update_dest_raw_single",
    "pagerank_calc_dest_single",
    "pagerank_calc_dest_raw_single",
    "pagerank_calc_diff_single"
};

typedef struct ocl_pagerank_kernel
{
    cl_kernel kernel;
    cl_command_queue queue;
    size_t work_group_size;
    size_t preferred_work_group_size_multiple;
} ocl_pagerank_kernel_t;

extern ocl_pagerank_kernel_t ocl_pagerank_kernels[STARPU_MAXOPENCLDEVS][E_PR_PROBLEM_KERNEL_MAX];

void ocl_pagerank_codelets_initialize(void);
void ocl_pagerank_codelets_finalize(void);

void ocl_pagerank_determine_work_groups(const ocl_pagerank_kernel_t *kernel, size_t *global_work_items, size_t *local_work_items);

cl_event ocl_pagerank_read_col(const int devid, pr_float *dst, const cl_mem src, const graph_size_t dst_offset, const graph_size_t dst_cols, const graph_size_t src_cols, const graph_size_t size, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_fill_arr(const int devid, const cl_mem arr, const pr_float val, const graph_size_t size, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_add_arr(const int devid, const cl_mem a, const cl_mem b, const graph_size_t size, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_sum_arr(const int devid, const cl_mem res, const cl_mem arr, const graph_size_t size, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_sum_arr_offset(const int devid, const cl_mem res, const cl_mem arr, const graph_size_t offset, const graph_size_t size, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_parallel_sum_arr(const int devid, const cl_mem res, const cl_mem arr, const cl_mem scr, const graph_size_t arr_size, const graph_size_t scr_size, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_parallel_sum_arr_offset(const int devid, const cl_mem res, const cl_mem arr, const cl_mem scr, const graph_size_t offset, const graph_size_t arr_size, const graph_size_t scr_size, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_baserank(const int devid, const cl_mem tmp, const cl_mem src, const cl_mem deg, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_baserank_redux(const int devid, const cl_mem tmp, const pr_float damping, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_update_rank_pull(const int devid, const cl_mem dst, const cl_mem src, const cl_mem rid, const cl_mem cid, const cl_mem deg, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_update_rank_push(const int devid, const cl_mem dst, const cl_mem src, const cl_mem rid, const cl_mem cid, const cl_mem deg, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_update_tmp(const int devid, const cl_mem tmp, const cl_mem src, const cl_mem deg, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_update_rank_tmp_pull(const int devid, const cl_mem dst, const cl_mem tmp, const cl_mem rid, const cl_mem cid, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_update_rank_tmp_push(const int devid, const cl_mem dst, const cl_mem tmp, const cl_mem rid, const cl_mem cid, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_update_rank_tmp_pull_warp(const int devid, const cl_mem dst, const cl_mem tmp, const cl_mem rid, const cl_mem cid, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_update_rank_tmp_push_warp(const int devid, const cl_mem dst, const cl_mem tmp, const cl_mem rid, const cl_mem cid, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_update_dest(const int devid, const cl_mem dst, const cl_mem base_rank, const pr_float damping, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_update_dest_raw(const int devid, const cl_mem dst, const pr_float base_rank, const pr_float damping, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_calc_dest(const int devid, const cl_mem dst, const cl_mem tmp, const cl_mem base_rank, const pr_float damping, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_calc_dest_raw(const int devid, const cl_mem dst, const cl_mem tmp, const pr_float base_rank, const pr_float damping, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);
cl_event ocl_pagerank_calc_diff(const int devid, const cl_mem tmp, const cl_mem src, const cl_mem dst, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list);

#ifdef __cplusplus
}
#endif