// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/spu_wrappers.h"
#include "alg/pagerank/cpu_codelets.h"
#include "alg/pagerank/ocl_codelets.h"
#include "alg/pagerank/cuda_codelets.h"

static inline void spu_pagerank_wait_for_opencl(const cl_event event)
{
    #ifdef STARPU_OPENCL_SYNC
        OPENCL_ASSERT(clWaitForEvents(1, &event));
        starpu_opencl_collect_stats(event);
    #endif

    clReleaseEvent(event);
}

static inline void spu_pagerank_wait_for_cuda(void)
{
    CUDA_ASSERT(cudaPeekAtLastError());
    #ifdef STARPU_CUDA_SYNC
        CUDA_ASSERT(cudaStreamSynchronize(starpu_cuda_get_local_stream()));
    #endif
}

void spu_pagerank_read_col_cpu(void *buffers[], void *args)
{
    const pr_float *src = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[0]);

    pr_float *dst = NULL;
    graph_size_t dst_offset = 0;
    graph_size_t dst_cols   = 0;
    graph_size_t src_cols   = 0;
    graph_size_t rows       = 0;
    starpu_codelet_unpack_args(args, &dst, &dst_offset, &dst_cols, &src_cols, &rows);

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= rows);

    cpu_pagerank_read_col(dst, src, dst_offset, dst_cols, src_cols, rows);
}

void spu_pagerank_read_col_ocl(void *buffers[], void *args)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);

    const cl_mem  src = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);

    pr_float *dst = NULL;
    graph_size_t dst_offset = 0;
    graph_size_t dst_cols   = 0;
    graph_size_t src_cols   = 0;
    graph_size_t rows       = 0;
    starpu_codelet_unpack_args(args, &dst, &dst_offset, &dst_cols, &src_cols, &rows);

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= rows);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_read_col(devid, dst, src, dst_offset, dst_cols, src_cols, rows, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_read_col_cuda(void *buffers[], void *args)
{
    const pr_float *src = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[0]);

    pr_float *dst = NULL;
    graph_size_t dst_offset = 0;
    graph_size_t dst_cols   = 0;
    graph_size_t src_cols   = 0;
    graph_size_t rows       = 0;
    starpu_codelet_unpack_args(args, &dst, &dst_offset, &dst_cols, &src_cols, &rows);

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= rows);

    cuda_pagerank_read_col(starpu_cuda_get_local_stream(), dst, src, dst_offset, dst_cols, src_cols, rows);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_fill_arr_cpu(void *buffers[], void *args)
{
          pr_float    *arr  = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[0]);
    const graph_size_t size = STARPU_VECTOR_GET_NX(buffers[0]);

    pr_float val = 0;
    starpu_codelet_unpack_args(args, &val);

    cpu_pagerank_fill_arr(arr, val, size);
}

void spu_pagerank_fill_arr_ocl(void *buffers[], void *args)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);

    const cl_mem       arr  = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
    const graph_size_t size = STARPU_VECTOR_GET_NX(buffers[0]);

    pr_float val = 0;
    starpu_codelet_unpack_args(args, &val);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_fill_arr(devid, arr, val, size, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_fill_arr_cuda(void *buffers[], void *args)
{
          pr_float    *arr  = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[0]);
    const graph_size_t size = STARPU_VECTOR_GET_NX(buffers[0]);

    pr_float val = 0;
    starpu_codelet_unpack_args(args, &val);

    cuda_pagerank_fill_arr(starpu_cuda_get_local_stream(), arr, val, size);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_redux_zero_single_cpu(void *buffers[], UNUSED void *args)
{
    pr_float *val = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[0]);

    cpu_pagerank_fill_arr(val, 0.0, 1);
}

void spu_pagerank_redux_zero_single_ocl(void *buffers[], UNUSED void *args)
{
    const cl_mem val = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[0]);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_fill_arr(devid, val, 0.0, 1, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_redux_zero_single_cuda(void *buffers[], UNUSED void *args)
{
    pr_float *val = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[0]);

    cuda_pagerank_fill_arr(starpu_cuda_get_local_stream(), val, 0.0, 1);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_redux_zero_cpu(void *buffers[], UNUSED void *args)
{
    pr_float val = 0.0;
    spu_pagerank_fill_arr_cpu(buffers, &val);
}

void spu_pagerank_redux_zero_ocl(void *buffers[], UNUSED void *args)
{
    pr_float val = 0.0;
    spu_pagerank_fill_arr_ocl(buffers, &val);
}

void spu_pagerank_redux_zero_cuda(void *buffers[], UNUSED void *args)
{
    pr_float val = 0.0;
    spu_pagerank_fill_arr_cuda(buffers, &val);
}

void spu_pagerank_redux_add_single_cpu(void *buffers[], UNUSED void *args)
{
          pr_float *a = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[0]);
    const pr_float *b = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[1]);

    cpu_pagerank_add_arr(a, b, 1);
}

void spu_pagerank_redux_add_single_ocl(void *buffers[], UNUSED void *args)
{
    const cl_mem a = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[0]);
    const cl_mem b = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[1]);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_add_arr(devid, a, b, 1, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_redux_add_single_cuda(void *buffers[], UNUSED void *args)
{
          pr_float *a = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[0]);
    const pr_float *b = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[1]);

    cuda_pagerank_add_arr(starpu_cuda_get_local_stream(), a, b, 1);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_redux_add_cpu(void *buffers[], UNUSED void *args)
{
          pr_float *a = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float *b = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[1]);

    const graph_size_t size = STARPU_VECTOR_GET_NX(buffers[0]);
    assert(STARPU_VECTOR_GET_NX(buffers[1]) == size);

    cpu_pagerank_add_arr(a, b, size);
}

void spu_pagerank_redux_add_ocl(void *buffers[], UNUSED void *args)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[1]) == 0);

    const cl_mem a = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
    const cl_mem b = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);

    const graph_size_t size = STARPU_VECTOR_GET_NX(buffers[0]);
    assert(STARPU_VECTOR_GET_NX(buffers[1]) == size);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_add_arr(devid, a, b, size, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_redux_add_cuda(void *buffers[], UNUSED void *args)
{
          pr_float *a = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float *b = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[1]);

    const graph_size_t size = STARPU_VECTOR_GET_NX(buffers[0]);
    assert(STARPU_VECTOR_GET_NX(buffers[1]) == size);

    cuda_pagerank_add_arr(starpu_cuda_get_local_stream(), a, b, size);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_redux_sum_cpu(void *buffers[], UNUSED void *args)
{
          pr_float *res = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[0]);
    const pr_float *arr = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[1]);

    const graph_size_t size = STARPU_VECTOR_GET_NX(buffers[1]);

    *res += cpu_pagerank_sum_arr(arr, size);
}

void spu_pagerank_redux_sum_ocl(void *buffers[], UNUSED void *args)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[1]) == 0);

    const cl_mem res = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[0]);
    const cl_mem arr = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);

    const graph_size_t size = STARPU_VECTOR_GET_NX(buffers[1]);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_sum_arr(devid, res, arr, size, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_redux_sum_cuda(void *buffers[], UNUSED void *args)
{
          pr_float *res = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[0]);
    const pr_float *arr = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[1]);

    const graph_size_t size = STARPU_VECTOR_GET_NX(buffers[1]);

    cuda_pagerank_sum_arr(starpu_cuda_get_local_stream(), res, arr, size);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_redux_parallel_sum_ocl(void *buffers[], UNUSED void *args)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[1]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[2]) == 0);

    const cl_mem res = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[0]);
    const cl_mem arr = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);
    const cl_mem scr = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[2]);

    const graph_size_t arr_size = STARPU_VECTOR_GET_NX(buffers[1]);
    const graph_size_t scr_size = STARPU_VECTOR_GET_NX(buffers[2]);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_parallel_sum_arr(devid, res, arr, scr, arr_size, scr_size, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_redux_parallel_sum_cuda(void *buffers[], UNUSED void *args)
{
          pr_float *res = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[0]);
    const pr_float *arr = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[1]);
          pr_float *scr = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[2]);

    const graph_size_t arr_size = STARPU_VECTOR_GET_NX(buffers[1]);
    const graph_size_t scr_size = STARPU_VECTOR_GET_NX(buffers[2]);

    cuda_pagerank_parallel_sum_arr(starpu_cuda_get_local_stream(), res, arr, scr, arr_size, scr_size);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_baserank_cpu(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *res = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *src = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *deg = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[0]);

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);
    assert(STARPU_VECTOR_GET_NX(buffers[2]) >= vcount);

    cpu_pagerank_baserank_mapped(res, src, deg, vcount);
}

void spu_pagerank_baserank_ocl(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[1]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[2]) == 0);

    const cl_mem res = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
    const cl_mem src = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);
    const cl_mem deg = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[2]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[0]);

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);
    assert(STARPU_VECTOR_GET_NX(buffers[2]) >= vcount);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_baserank(devid, res, src, deg, vcount, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_baserank_cuda(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *res = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *src = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *deg = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[0]);

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);
    assert(STARPU_VECTOR_GET_NX(buffers[2]) >= vcount);

    cuda_pagerank_baserank(starpu_cuda_get_local_stream(), res, src, deg, vcount);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_baserank_redux_cpu(void *buffers[],  spu_pr_problem_t *problem)
{
    pr_float *redux = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[0]);

    const graph_size_t vcount  = problem->graph->vcount;
    const pr_float     damping = problem->options->damping;

    *redux = cpu_pagerank_baserank_redux(*redux, damping, vcount);
}

void spu_pagerank_baserank_redux_ocl(void *buffers[], spu_pr_problem_t *problem)
{
    cl_mem redux = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[0]);

    const graph_size_t vcount  = problem->graph->vcount;
    const pr_float     damping = problem->options->damping;

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_baserank_redux(devid, redux, damping, vcount, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_baserank_redux_cuda(void *buffers[],  spu_pr_problem_t *problem)
{
    pr_float *redux = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[0]);

    const graph_size_t vcount  = problem->graph->vcount;
    const pr_float     damping = problem->options->damping;

    cuda_pagerank_baserank_redux(starpu_cuda_get_local_stream(), redux, damping, vcount);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_update_rank_pull_cpu(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *dst = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *src = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *deg = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
    const graph_size_t *rid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[3]);
    const graph_size_t *cid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[4]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[3]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= vcount);

    cpu_pagerank_update_rank_pull(dst, src, rid, cid, deg, vcount);
}

void spu_pagerank_update_rank_pull_ocl(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[1]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[2]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[3]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[4]) == 0);

    const cl_mem dst = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
    const cl_mem src = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);
    const cl_mem deg = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[2]);
    const cl_mem rid = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[3]);
    const cl_mem cid = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[4]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[3]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= vcount);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_update_rank_pull(devid, dst, src, rid, cid, deg, vcount, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_update_rank_pull_cuda(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *dst = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *src = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *deg = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
    const graph_size_t *rid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[3]);
    const graph_size_t *cid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[4]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[3]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= vcount);

    cuda_pagerank_update_rank_pull(starpu_cuda_get_local_stream(), dst, src, rid, cid, deg, vcount);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_update_rank_push_cpu(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *dst = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *src = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *deg = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
    const graph_size_t *rid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[3]);
    const graph_size_t *cid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[4]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[3]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);

    cpu_pagerank_update_rank_push(dst, src, rid, cid, deg, vcount);
}

void spu_pagerank_update_rank_push_ocl(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[1]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[2]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[3]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[4]) == 0);

    const cl_mem dst = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
    const cl_mem src = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);
    const cl_mem deg = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[2]);
    const cl_mem rid = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[3]);
    const cl_mem cid = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[4]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[3]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_update_rank_push(devid, dst, src, rid, cid, deg, vcount, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_update_rank_push_cuda(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *dst = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *src = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *deg = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
    const graph_size_t *rid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[3]);
    const graph_size_t *cid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[4]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[3]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);

    cuda_pagerank_update_rank_push(starpu_cuda_get_local_stream(), dst, src, rid, cid, deg, vcount);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_update_tmp_cpu(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *tmp = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *src = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *deg = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[0]);

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);
    assert(STARPU_VECTOR_GET_NX(buffers[2]) >= vcount);

    cpu_pagerank_update_tmp(tmp, src, deg, vcount);
}

void spu_pagerank_update_tmp_ocl(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[1]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[2]) == 0);

    const cl_mem tmp = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
    const cl_mem src = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);
    const cl_mem deg = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[2]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[0]);

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);
    assert(STARPU_VECTOR_GET_NX(buffers[2]) >= vcount);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_update_tmp(devid, tmp, src, deg, vcount, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_update_tmp_cuda(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *tmp = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *src = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *deg = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[0]);

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);
    assert(STARPU_VECTOR_GET_NX(buffers[2]) >= vcount);

    cuda_pagerank_update_tmp(starpu_cuda_get_local_stream(), tmp, src, deg, vcount);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_update_rank_tmp_pull_cpu(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *dst = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *tmp = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *rid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
    const graph_size_t *cid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[3]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[2]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= vcount);

    cpu_pagerank_update_rank_tmp_pull(dst, tmp, rid, cid, vcount);
}

void spu_pagerank_update_rank_tmp_pull_ocl(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[1]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[2]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[3]) == 0);

    const cl_mem dst = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
    const cl_mem tmp = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);
    const cl_mem rid = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[2]);
    const cl_mem cid = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[3]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[2]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= vcount);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_update_rank_tmp_pull(devid, dst, tmp, rid, cid, vcount, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_update_rank_tmp_pull_cuda(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *dst = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *tmp = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *rid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
    const graph_size_t *cid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[3]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[2]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= vcount);

    cuda_pagerank_update_rank_tmp_pull(starpu_cuda_get_local_stream(), dst, tmp, rid, cid, vcount);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_update_rank_tmp_push_cpu(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *dst = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *tmp = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *rid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
    const graph_size_t *cid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[3]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[2]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);

    cpu_pagerank_update_rank_tmp_push(dst, tmp, rid, cid, vcount);
}

void spu_pagerank_update_rank_tmp_push_ocl(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[1]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[2]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[3]) == 0);

    const cl_mem dst = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
    const cl_mem tmp = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);
    const cl_mem rid = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[2]);
    const cl_mem cid = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[3]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[2]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_update_rank_tmp_push(devid, dst, tmp, rid, cid, vcount, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_update_rank_tmp_push_cuda(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *dst = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *tmp = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *rid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
    const graph_size_t *cid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[3]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[2]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);

    cuda_pagerank_update_rank_tmp_push(starpu_cuda_get_local_stream(), dst, tmp, rid, cid, vcount);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_update_rank_tmp_pull_warp_ocl(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[1]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[2]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[3]) == 0);

    const cl_mem dst = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
    const cl_mem tmp = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);
    const cl_mem rid = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[2]);
    const cl_mem cid = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[3]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[2]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= vcount);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_update_rank_tmp_pull_warp(devid, dst, tmp, rid, cid, vcount, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_update_rank_tmp_pull_warp_cuda(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *dst = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *tmp = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *rid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
    const graph_size_t *cid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[3]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[2]) - 1;
    const graph_size_t ecount = STARPU_VECTOR_GET_NX(buffers[3]);

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= vcount);

    cuda_pagerank_update_rank_tmp_pull_warp(starpu_cuda_get_local_stream(), ecount / vcount, dst, tmp, rid, cid, vcount);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_update_rank_tmp_pull_dyn_cuda(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *dst = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *tmp = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *rid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
    const graph_size_t *cid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[3]);
          graph_size_t *scr = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[4]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[2]) - 1;
    const graph_size_t ecount = STARPU_VECTOR_GET_NX(buffers[3]);

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= vcount);

    cuda_pagerank_update_rank_tmp_pull_dyn(starpu_cuda_get_local_stream(), ecount / vcount, scr, dst, tmp, rid, cid, vcount);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_update_rank_tmp_push_warp_ocl(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[1]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[2]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[3]) == 0);

    const cl_mem dst = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
    const cl_mem tmp = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);
    const cl_mem rid = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[2]);
    const cl_mem cid = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[3]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[2]) - 1;

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_update_rank_tmp_push_warp(devid, dst, tmp, rid, cid, vcount, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_update_rank_tmp_push_warp_cuda(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *dst = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *tmp = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *rid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
    const graph_size_t *cid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[3]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[2]) - 1;
    const graph_size_t ecount = STARPU_VECTOR_GET_NX(buffers[3]);

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);

    cuda_pagerank_update_rank_tmp_push_warp(starpu_cuda_get_local_stream(), ecount / vcount, dst, tmp, rid, cid, vcount);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_update_rank_tmp_push_dyn_cuda(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float     *dst = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[0]);
    const pr_float     *tmp = (pr_float*)     STARPU_VECTOR_GET_PTR(buffers[1]);
    const graph_size_t *rid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
    const graph_size_t *cid = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[3]);
          graph_size_t *scr = (graph_size_t*) STARPU_VECTOR_GET_PTR(buffers[4]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[2]) - 1;
    const graph_size_t ecount = STARPU_VECTOR_GET_NX(buffers[3]);

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);

    cuda_pagerank_update_rank_tmp_push_dyn(starpu_cuda_get_local_stream(), ecount / vcount, scr, dst, tmp, rid, cid, vcount);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_update_dest_cpu(void *buffers[], spu_pr_problem_t *problem)
{
    pr_float *dst = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[0]);
    pr_float *rnk = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[1]);

    const pr_float      damping = problem->options->damping;
    const graph_size_t  vcount  = STARPU_VECTOR_GET_NX(buffers[0]);

    cpu_pagerank_update_dest(dst, *rnk, damping, vcount);
}

void spu_pagerank_update_dest_ocl(void *buffers[], spu_pr_problem_t *problem)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);

    const cl_mem dst = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
    const cl_mem rnk = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[1]);

    const pr_float      damping = problem->options->damping;
    const graph_size_t  vcount  = STARPU_VECTOR_GET_NX(buffers[0]);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_update_dest(devid, dst, rnk, damping, vcount, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_update_dest_cuda(void *buffers[], spu_pr_problem_t *problem)
{
    pr_float *dst = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[0]);
    pr_float *rnk = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[1]);

    const pr_float      damping = problem->options->damping;
    const graph_size_t  vcount  = STARPU_VECTOR_GET_NX(buffers[0]);

    cuda_pagerank_update_dest(starpu_cuda_get_local_stream(), dst, rnk, damping, vcount);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_calc_dest_cpu(void *buffers[], spu_pr_problem_t *problem)
{
    pr_float *dst = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[0]);
    pr_float *tmp = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[1]);
    pr_float *rnk = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[2]);

    const pr_float      damping = problem->options->damping;
    const graph_size_t  vcount  = STARPU_VECTOR_GET_NX(buffers[1]);

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= vcount);

    cpu_pagerank_calc_dest(dst, tmp, *rnk, damping, vcount);
}

void spu_pagerank_calc_dest_ocl(void *buffers[], spu_pr_problem_t *problem)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[1]) == 0);

    const cl_mem dst = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
    const cl_mem tmp = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);
    const cl_mem rnk = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[2]);

    const pr_float      damping = problem->options->damping;
    const graph_size_t  vcount  = STARPU_VECTOR_GET_NX(buffers[1]);

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= vcount);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_calc_dest(devid, dst, tmp, rnk, damping, vcount, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_calc_dest_cuda(void *buffers[], spu_pr_problem_t *problem)
{
    pr_float *dst = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[0]);
    pr_float *tmp = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[1]);
    pr_float *rnk = (pr_float*) STARPU_VARIABLE_GET_PTR(buffers[2]);

    const pr_float      damping = problem->options->damping;
    const graph_size_t  vcount  = STARPU_VECTOR_GET_NX(buffers[1]);

    assert(STARPU_VECTOR_GET_NX(buffers[0]) >= vcount);

    cuda_pagerank_calc_dest(starpu_cuda_get_local_stream(), dst, tmp, rnk, damping, vcount);
    spu_pagerank_wait_for_cuda();
}

void spu_pagerank_calc_diff_cpu(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float *dif = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[0]);
          pr_float *dst = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[1]);
    const pr_float *src = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[2]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[0]);

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);
    assert(STARPU_VECTOR_GET_NX(buffers[2]) >= vcount);

    cpu_pagerank_calc_diff_mapped(dif, src, dst, vcount);
}

void spu_pagerank_calc_diff_ocl(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
    assert(STARPU_VECTOR_GET_OFFSET(buffers[0]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[1]) == 0);
    assert(STARPU_VECTOR_GET_OFFSET(buffers[2]) == 0);

    const cl_mem dif = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
    const cl_mem dst = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);
    const cl_mem src = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[2]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[0]);

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);
    assert(STARPU_VECTOR_GET_NX(buffers[2]) >= vcount);

    const int      devid = starpu_worker_get_devid(starpu_worker_get_id());
    const cl_event event = ocl_pagerank_calc_diff(devid, dif, src, dst, vcount, 0, NULL);

    spu_pagerank_wait_for_opencl(event);
}

void spu_pagerank_calc_diff_cuda(void *buffers[], UNUSED spu_pr_problem_t *problem)
{
          pr_float *dif = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[0]);
          pr_float *dst = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[1]);
    const pr_float *src = (pr_float*) STARPU_VECTOR_GET_PTR(buffers[2]);

    const graph_size_t vcount = STARPU_VECTOR_GET_NX(buffers[0]);

    assert(STARPU_VECTOR_GET_NX(buffers[1]) >= vcount);
    assert(STARPU_VECTOR_GET_NX(buffers[2]) >= vcount);

    cuda_pagerank_calc_diff(starpu_cuda_get_local_stream(), dif, src, dst, vcount);
    spu_pagerank_wait_for_cuda();
}
