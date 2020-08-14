// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/cpu_codelets.h"
#include "alg/pagerank/ocl_codelets.h"
#include "util/math.h"

#define BENCH_INIT_TMPV(IDX) \
    cl_mem v ## IDX = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(pr_float) * graph->vcount, NULL, &err); OPENCL_ASSERT(err); \
    clReleaseEvent(ocl_pagerank_fill_arr(options->devid, v ## IDX, 0, graph->vcount, 0, NULL));
#define BENCH_INIT_TMPE(IDX) \
    cl_mem e ## IDX = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(pr_float) * graph->ecount, NULL, &err); OPENCL_ASSERT(err); \
    clReleaseEvent(ocl_pagerank_fill_arr(options->devid, e ## IDX, 0, graph->ecount, 0, NULL));
#define BENCH_FREE_TMPV(IDX) OPENCL_ASSERT(clReleaseMemObject(v ## IDX));
#define BENCH_FREE_TMPE(IDX) OPENCL_ASSERT(clReleaseMemObject(e ## IDX));

#define BENCH(NAME,STAGE,DEG,TMPV,TMPE,CODE) \
uint32_t bench_ ## NAME(const pr_csr_graph_t *graph, pagerank_options_t *options) \
{ \
    assert(graph != NULL); \
    cl_int err; \
    cl_context context; \
    cl_command_queue queue; \
    starpu_opencl_get_context(options->devid, &context); \
    starpu_opencl_get_queue(options->devid, &queue); \
    clFinish(queue); \
    PAGERANK_TIME_START(INIT) \
    cl_mem scr  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(pr_float) * PAGERANK_SCRATCH_SIZE, NULL, &err); OPENCL_ASSERT(err); \
    clReleaseEvent(ocl_pagerank_fill_arr(options->devid, scr, 0, PAGERANK_SCRATCH_SIZE, 0, NULL)); \
    REPEAT(BENCH_INIT_TMPV,TMPV) \
    REPEAT(BENCH_INIT_TMPE,TMPE) \
    clFinish(queue); \
    PAGERANK_TIME_STOP(INIT) \
    PAGERANK_TIME_START(TRANSFER) \
    cl_mem row = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(*graph->row_idx) * (graph->vcount + 1), graph->row_idx, &err); OPENCL_ASSERT(err); \
    cl_mem col = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(*graph->col_idx) * graph->ecount,       graph->col_idx, &err); OPENCL_ASSERT(err); \
    cl_mem deg = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(*graph->DEG)     * graph->vcount,       graph->DEG,     &err); OPENCL_ASSERT(err); \
    clFinish(queue); \
    PAGERANK_TIME_STOP(TRANSFER) \
    for (uint32_t it = 0; it < options->min_iterations; it++) \
    { \
        cl_event start, end; \
        CODE \
        OPENCL_ASSERT(clWaitForEvents(1, &end)); \
        cl_ulong time_start, time_end; \
        OPENCL_ASSERT(clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL)); \
        OPENCL_ASSERT(clGetEventProfilingInfo(end,   CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &time_end,   NULL));  \
        options->stage_time[E_PR_STAGE_ ## STAGE] += (time_diff_t)1.0e-9 * (time_end - time_start); \
        if (start != end) \
            clReleaseEvent(start); \
        clReleaseEvent(end); \
    } \
    OPENCL_ASSERT(clReleaseMemObject(deg)); \
    OPENCL_ASSERT(clReleaseMemObject(col)); \
    OPENCL_ASSERT(clReleaseMemObject(row)); \
    OPENCL_ASSERT(clReleaseMemObject(scr)); \
    REPEAT(BENCH_FREE_TMPE,TMPE) \
    REPEAT(BENCH_FREE_TMPV,TMPV) \
    return options->min_iterations; \
}

BENCH(fill_ocl_default, DIFF, deg_o, 1, 0,
    start = end = ocl_pagerank_fill_arr(options->devid, v0, 1.0 / graph->vcount, graph->vcount, 0, NULL);
)

BENCH(asum_ocl_default, DIFF, deg_o, 1, 0,
    start = end = ocl_pagerank_sum_arr_offset(options->devid, scr, v0, 0, graph->vcount, 0, NULL);
)

BENCH(asum_ocl_parallel, DIFF, deg_o, 1, 0,
    start = end = ocl_pagerank_parallel_sum_arr_offset(options->devid, scr, v0, scr, 0, graph->vcount, PAGERANK_SCRATCH_SIZE, 0, NULL);
)

BENCH(base_ocl_mapped, BASERANK, deg_o, 2, 0,
    start = end = ocl_pagerank_baserank(options->devid, v1, v0, deg, graph->vcount, 0, NULL);
)

BENCH(diff_ocl_mapdef, DIFF, deg_o, 3, 0,
    start = ocl_pagerank_calc_diff(options->devid, v2, v0, v1, graph->vcount, 0, NULL);
     end  = ocl_pagerank_sum_arr(options->devid, scr, v2, graph->vcount, 1, &start);
)

BENCH(diff_ocl_mappar, DIFF, deg_o, 3, 0,
    start = ocl_pagerank_calc_diff(options->devid, v2, v0, v1, graph->vcount, 0, NULL);
    end   = ocl_pagerank_parallel_sum_arr(options->devid, scr, scr, v2, graph->vcount, PAGERANK_SCRATCH_SIZE, 1, &start);
)

BENCH(update_csr_ocl_default, UPDATE, deg_o, 2, 0,
    start = end = ocl_pagerank_update_rank_push(options->devid, v1, v0, row, col, deg, graph->vcount, 0, NULL);
)

BENCH(update_csc_ocl_default, UPDATE, deg_i, 2, 0,
    start = end = ocl_pagerank_update_rank_pull(options->devid, v1, v0, row, col, deg, graph->vcount, 0, NULL);
)

BENCH(update_csr_ocl_stepped, UPDATE, deg_o, 3, 0,
    start = ocl_pagerank_update_tmp(options->devid, v2, v0, deg, graph->vcount, 0, NULL);
    end   = ocl_pagerank_update_rank_tmp_push(options->devid, v1, v2, row, col, graph->vcount, 1, &start);
)

BENCH(update_csc_ocl_stepped, UPDATE, deg_i, 3, 0,
    start = ocl_pagerank_update_tmp(options->devid, v2, v0, deg, graph->vcount, 0, NULL);
    end   = ocl_pagerank_update_rank_tmp_pull(options->devid, v1, v2, row, col, graph->vcount, 1, &start);
)

BENCH(update_csr_ocl_warp, UPDATE, deg_o, 3, 0,
    start = ocl_pagerank_update_tmp(options->devid, v2, v0, deg, graph->vcount, 0, NULL);
    end   = ocl_pagerank_update_rank_tmp_push_warp(options->devid, v1, v2, row, col, graph->vcount, 1, &start);
)

BENCH(update_csc_ocl_warp, UPDATE, deg_i, 3, 0,
    start = ocl_pagerank_update_tmp(options->devid, v2, v0, deg, graph->vcount, 0, NULL);
     end  = ocl_pagerank_update_rank_tmp_pull_warp(options->devid, v1, v2, row, col, graph->vcount, 1, &start);
)