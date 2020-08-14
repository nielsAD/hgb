// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/cpu_codelets.h"
#include "alg/pagerank/cuda_codelets.h"
#include "util/math.h"

#define BENCH_INIT_TMPV(IDX) \
    pr_float *restrict v ## IDX; CUDA_ASSERT(cudaMalloc((void**) &v ## IDX, sizeof(pr_float) * graph->vcount)); \
    CUDA_ASSERT(cudaMemset(v ## IDX, 0, sizeof(pr_float) * graph->vcount));
#define BENCH_INIT_TMPE(IDX) \
    pr_float *restrict e ## IDX; CUDA_ASSERT(cudaMalloc((void**) &e ## IDX, sizeof(pr_float) * graph->ecount)); \
    CUDA_ASSERT(cudaMemset(e ## IDX, 0, sizeof(pr_float) * graph->ecount));
#define BENCH_FREE_TMPV(IDX) CUDA_ASSERT(cudaFree(v ## IDX));
#define BENCH_FREE_TMPE(IDX) CUDA_ASSERT(cudaFree(e ## IDX));

#define BENCH(NAME,STAGE,DEG,TMPV,TMPE,CODE) \
uint32_t bench_ ## NAME(const pr_csr_graph_t *graph, pagerank_options_t *options) \
{ \
    assert(graph != NULL); \
    CUDA_ASSERT(cudaStreamSynchronize(NULL)); \
    PAGERANK_TIME_START(INIT) \
    cusparseMatDescr_t mat_descr; CUDA_ASSERT(cusparseCreateMatDescr(&mat_descr)); \
    graph_size_t *restrict row; CUDA_ASSERT(cudaMalloc((void**) &row, sizeof(*graph->row_idx) * (graph->vcount+1))); \
    graph_size_t *restrict col; CUDA_ASSERT(cudaMalloc((void**) &col, sizeof(*graph->col_idx) * graph->ecount)); \
    graph_size_t *restrict deg; CUDA_ASSERT(cudaMalloc((void**) &deg, sizeof(*graph->DEG)     * graph->vcount)); \
    graph_size_t *restrict scru; CUDA_ASSERT(cudaMalloc((void**) &scru, sizeof(graph_size_t) * PAGERANK_SCRATCH_SIZE)); \
    pr_float     *restrict scrf; CUDA_ASSERT(cudaMalloc((void**) &scrf, sizeof(pr_float)     * PAGERANK_SCRATCH_SIZE)); \
    CUDA_ASSERT(cudaMemset(scrf, 0, sizeof(pr_float) * PAGERANK_SCRATCH_SIZE)); \
    REPEAT(BENCH_INIT_TMPV,TMPV) \
    REPEAT(BENCH_INIT_TMPE,TMPE) \
    CUDA_ASSERT(cudaStreamSynchronize(NULL)); \
    PAGERANK_TIME_STOP(INIT) \
    PAGERANK_TIME_START(TRANSFER) \
    CUDA_ASSERT(cudaMemcpy(row, graph->row_idx, sizeof(graph_size_t) * (graph->vcount + 1), cudaMemcpyHostToDevice)); \
    CUDA_ASSERT(cudaMemcpy(col, graph->col_idx, sizeof(graph_size_t) *  graph->ecount,      cudaMemcpyHostToDevice)); \
    CUDA_ASSERT(cudaMemcpy(deg, graph->DEG,     sizeof(graph_size_t) *  graph->vcount,      cudaMemcpyHostToDevice)); \
    CUDA_ASSERT(cudaStreamSynchronize(NULL)); \
    PAGERANK_TIME_STOP(TRANSFER) \
    for (uint32_t it = 0; it < options->min_iterations; it++) \
    { \
        cudaEvent_t start; CUDA_ASSERT(cudaEventCreate(&start)); CUDA_ASSERT(cudaEventRecord(start, NULL)); \
        CODE \
        cudaEvent_t end; CUDA_ASSERT(cudaEventCreate(&end)); CUDA_ASSERT(cudaEventRecord(end, NULL)); \
        CUDA_ASSERT(cudaEventSynchronize(end)); \
        float time_ms; \
        CUDA_ASSERT(cudaEventElapsedTime(&time_ms, start, end)); \
        options->stage_time[E_PR_STAGE_ ## STAGE] += (time_diff_t)1.0e-3 * time_ms; \
        cudaEventDestroy(start); \
        cudaEventDestroy(end); \
    } \
    CUDA_ASSERT(cudaFree(deg)); \
    CUDA_ASSERT(cudaFree(col)); \
    CUDA_ASSERT(cudaFree(row)); \
    CUDA_ASSERT(cudaFree(scrf)); \
    CUDA_ASSERT(cudaFree(scru)); \
    REPEAT(BENCH_FREE_TMPE,TMPE) \
    REPEAT(BENCH_FREE_TMPV,TMPV) \
    CUDA_ASSERT(cusparseDestroyMatDescr(mat_descr)); \
    return options->min_iterations; \
}

BENCH(fill_cud_default, DIFF, deg_o, 1, 0,
    cuda_pagerank_fill_arr(NULL, v0, 1.0 / graph->vcount, graph->vcount);
)

BENCH(asum_cud_default, DIFF, deg_o, 1, 0,
    cuda_pagerank_sum_arr_offset(NULL, scrf, v0, 0, graph->vcount);
)

BENCH(asum_cud_parallel, DIFF, deg_o, 1, 0,
    cuda_pagerank_parallel_sum_arr_offset(NULL, scrf, v0, scrf, 0, graph->vcount, PAGERANK_SCRATCH_SIZE);
)

BENCH(asum_cud_lib, DIFF, deg_o, 1, 0,
    pr_float tmpf;
    CUDA_ASSERT(cublasSasum(handle_cublas, graph->vcount, v0, 1, &tmpf));
)

BENCH(base_cud_mapped, BASERANK, deg_o, 2, 0,
    cuda_pagerank_baserank(NULL, v1, v0, deg, graph->vcount);
)

BENCH(diff_cud_mapdef, DIFF, deg_o, 3, 0,
    cuda_pagerank_calc_diff(NULL, v2, v0, v1, graph->vcount);
    cuda_pagerank_sum_arr(NULL, scrf, v2, graph->vcount);
)

BENCH(diff_cud_mappar, DIFF, deg_o, 3, 0,
    cuda_pagerank_calc_diff(NULL, v2, v0, v1, graph->vcount);
    cuda_pagerank_parallel_sum_arr(NULL, scrf, scrf, v2, graph->vcount, PAGERANK_SCRATCH_SIZE);
)

BENCH(diff_cud_lib, DIFF, deg_o, 3, 0,
    pr_float tmpf;
    cuda_pagerank_calc_diff(NULL, v2, v0, v1, graph->vcount);
    CUDA_ASSERT(cublasSasum(handle_cublas, graph->vcount, v0, 1, &tmpf));
)

BENCH(update_csr_cud_default, UPDATE, deg_o, 2, 0,
    cuda_pagerank_update_rank_push(NULL, v1, v0, row, col, deg, graph->vcount);
)

BENCH(update_csc_cud_default, UPDATE, deg_i, 2, 0,
    cuda_pagerank_update_rank_pull(NULL, v1, v0, row, col, deg, graph->vcount);
)

BENCH(update_csr_cud_stepped, UPDATE, deg_o, 3, 0,
    cuda_pagerank_update_tmp(NULL, v2, v0, deg, graph->vcount);
    cuda_pagerank_update_rank_tmp_push(NULL, v1, v2, row, col, graph->vcount);
)

BENCH(update_csc_cud_stepped, UPDATE, deg_i, 3, 0,
    cuda_pagerank_update_tmp(NULL, v2, v0, deg, graph->vcount);
    cuda_pagerank_update_rank_tmp_pull(NULL, v1, v2, row, col, graph->vcount);
)

BENCH(update_csr_cud_warp, UPDATE, deg_o, 3, 0,
    cuda_pagerank_update_tmp(NULL, v2, v0, deg, graph->vcount);
    cuda_pagerank_update_rank_tmp_push_warp(NULL, graph->ecount / graph->vcount, v1, v2, row, col, graph->vcount);
)

BENCH(update_csc_cud_warp, UPDATE, deg_i, 3, 0,
    cuda_pagerank_update_tmp(NULL, v2, v0, deg, graph->vcount);
    cuda_pagerank_update_rank_tmp_pull_warp(NULL, graph->ecount / graph->vcount, v1, v2, row, col, graph->vcount);
)

BENCH(update_csr_cud_dyn, UPDATE, deg_o, 3, 0,
    cuda_pagerank_update_tmp(NULL, v2, v0, deg, graph->vcount);
    cuda_pagerank_update_rank_tmp_push_dyn(NULL, graph->ecount / graph->vcount, scru, v1, v2, row, col, graph->vcount);
)

BENCH(update_csc_cud_dyn, UPDATE, deg_i, 3, 0,
    cuda_pagerank_update_tmp(NULL, v2, v0, deg, graph->vcount);
    cuda_pagerank_update_rank_tmp_pull_dyn(NULL, graph->ecount / graph->vcount, scru, v1, v2, row, col, graph->vcount);
)

BENCH(update_csr_cud_lib, UPDATE, deg_o, 2, 1,
    static const pr_float zero = 0.0;
    static const pr_float one  = 1.0;

    CUDA_ASSERT(cusparseScsrmv(handle_cusparse,
        CUSPARSE_OPERATION_TRANSPOSE,
        graph->vcount, graph->vcount, graph->ecount,
        &one, mat_descr,
        e0, (int*)row, (int*)col,
        v0, &zero, v1
    ));
)

BENCH(update_csc_cud_lib, UPDATE, deg_i, 2, 1,
    static const pr_float zero = 0.0;
    static const pr_float one  = 1.0;

    CUDA_ASSERT(cusparseScsrmv(handle_cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        graph->vcount, graph->vcount, graph->ecount,
        &one, mat_descr,
        e0, (int*)row, (int*)col,
        v0, &zero, v1
    ));
)