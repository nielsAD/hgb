// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/cpu_codelets.h"
#include "util/math.h"
#include "util/memory.h"
#include "util/mkl.h"

#define BENCH_INIT_TMPV(IDX) pr_float *restrict v ## IDX = memory_talloc(pr_float, graph->vcount); memset(v ## IDX, 0, sizeof(pr_float) * graph->vcount);
#define BENCH_INIT_TMPE(IDX) pr_float *restrict e ## IDX = memory_talloc(pr_float, graph->ecount); memset(e ## IDX, 0, sizeof(pr_float) * graph->ecount);
#define BENCH_FREE_TMPV(IDX) memory_free((void*)v ## IDX);
#define BENCH_FREE_TMPE(IDX) memory_free((void*)e ## IDX);

#define BENCH(NAME,STAGE,TMPV,TMPE,CODE) \
uint32_t bench_ ## NAME(const pr_csr_graph_t *graph, pagerank_options_t *options) \
{ \
    assert(graph != NULL); \
    PAGERANK_TIME_START(INIT) \
    REPEAT(BENCH_INIT_TMPV,TMPV) \
    REPEAT(BENCH_INIT_TMPE,TMPE) \
    UNUSED volatile pr_float tmpf; \
    PAGERANK_TIME_STOP(INIT) \
    PAGERANK_TIME_START(STAGE) \
    for (uint32_t it = 0; it < options->min_iterations; it++) \
    { \
        CODE \
    } \
    PAGERANK_TIME_STOP(STAGE) \
    REPEAT(BENCH_FREE_TMPE,TMPE) \
    REPEAT(BENCH_FREE_TMPV,TMPV) \
    return options->min_iterations; \
}

BENCH(fill_cpu_default, DIFF, 1, 0,
    cpu_pagerank_fill_arr(v0, 1.0 / graph->vcount, graph->vcount);
)

BENCH(asum_cpu_default, DIFF, 1, 0,
    tmpf = cpu_pagerank_sum_arr(v0, graph->vcount);
)

BENCH(asum_cpu_lib, DIFF, 1, 0,
    tmpf = cblas_sasum(graph->vcount, options->result, 1);
)

BENCH(base_cpu_default, BASERANK, 1, 0,
    tmpf = cpu_pagerank_baserank(v0, graph->deg_o, graph->vcount);
)

BENCH(base_cpu_mapped, BASERANK, 2, 0,
    cpu_pagerank_baserank_mapped(v1, v0, graph->deg_o, graph->vcount);
    tmpf = cpu_pagerank_sum_arr(v1, graph->vcount);
)

BENCH(diff_cpu_default, DIFF, 2, 0,
    tmpf = cpu_pagerank_calc_diff(v0, v1, graph->vcount);
)

BENCH(diff_cpu_mapped, DIFF, 3, 0,
    cpu_pagerank_calc_diff_mapped(v2, v0, v1, graph->vcount);
    tmpf = cpu_pagerank_sum_arr(v2, graph->vcount);
)

BENCH(diff_cpu_lib, DIFF, 3, 0,
    vmsSub(graph->vcount, v1, v0, v2, VML_EP | VML_FTZDAZ_ON);
    vmsAbs(graph->vcount, v2, v2, VML_EP | VML_FTZDAZ_ON);
    tmpf = cblas_sasum(graph->vcount, v2, 1);
)

BENCH(update_csr_cpu_default, UPDATE, 2, 0,
    cpu_pagerank_update_rank_push(v1, v0, graph->row_idx, graph->col_idx, graph->deg_o, graph->vcount);
)

BENCH(update_csc_cpu_default, UPDATE, 2, 0,
    cpu_pagerank_update_rank_pull(v1, v0, graph->row_idx, graph->col_idx, graph->deg_i, graph->vcount);
)

BENCH(update_csr_cpu_stepped, UPDATE, 3, 0,
    cpu_pagerank_update_tmp(v2, v0, graph->deg_o, graph->vcount);
    cpu_pagerank_update_rank_tmp_push(v1, v2, graph->row_idx, graph->col_idx, graph->vcount);
)

BENCH(update_csc_cpu_stepped, UPDATE, 3, 0,
    cpu_pagerank_update_tmp(v2, v0, graph->deg_i, graph->vcount);
    cpu_pagerank_update_rank_tmp_pull(v1, v2, graph->row_idx, graph->col_idx, graph->vcount);
)

BENCH(update_csr_cpu_lib, UPDATE, 2, 1,
    mkl_cspblas_scsrgemv("t", (MKL_INT*)&graph->vcount, e0, (MKL_INT*)graph->row_idx, (MKL_INT*)graph->col_idx, v0, v1);
)

BENCH(update_csc_cpu_lib, UPDATE, 2, 1,
    mkl_cspblas_scsrgemv("n", (MKL_INT*)&graph->vcount, e0, (MKL_INT*)graph->row_idx, (MKL_INT*)graph->col_idx, v0, v1);
)
