// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/cpu_codelets.h"
#include "util/math.h"
#include "util/memory.h"
#include "util/mkl.h"

uint32_t pagerank_csc_mkl_lib(const pr_csc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    PAGERANK_TIME_START(INIT)

    const graph_size_t vcount = graph->vcount;

    pr_float *restrict src = memory_talloc(pr_float, vcount);
    pr_float *restrict dst = memory_talloc(pr_float, vcount);
    pr_float *restrict tmp = memory_talloc(pr_float, vcount);
    pr_float *restrict val = memory_talloc(pr_float, graph->ecount);

    cpu_pagerank_fill_arr(dst, 1.0 / vcount, vcount);

    csr_forall_edges_par(e, efr, eto, graph, simd)
    {
        val[e] = 1.0 / graph->deg_i[eto];
    }
    PAGERANK_TIME_STOP(INIT)

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            SWAP_VALUES(src, dst);

            PAGERANK_TIME_START(BASERANK)
            cpu_pagerank_baserank_mapped(tmp, src, graph->deg_i, vcount);

            pr_float base_rank = cpu_pagerank_baserank_redux(
                cblas_sasum(vcount, tmp, 1),
                options->damping,
                vcount
            );
            PAGERANK_TIME_STOP(BASERANK)

            PAGERANK_TIME_START(UPDATE)
            mkl_cspblas_scsrgemv("n", (MKL_INT*)&vcount, val, (MKL_INT*)graph->row_idx, (MKL_INT*)graph->col_idx, src, dst);
            cpu_pagerank_update_dest(dst, base_rank, options->damping, vcount);
            PAGERANK_TIME_STOP(UPDATE)
        }

        PAGERANK_TIME_START(DIFF)
        vsSub(vcount, dst, src, tmp);
        vsAbs(vcount, tmp, tmp);
        diff = cblas_sasum(vcount, tmp, 1);
        PAGERANK_TIME_STOP(DIFF)

        iterations += options->local_iterations;
    }

    if (options->result != NULL)
    {
        PAGERANK_TIME_START(TRANSFER)
        cblas_scopy(vcount, dst, 1, options->result, 1);
        PAGERANK_TIME_STOP(TRANSFER)
    }

    memory_free((void*)val);
    memory_free((void*)tmp);
    memory_free((void*)src);
    memory_free((void*)dst);

    return iterations;
}

uint32_t pagerank_bcsc_ref_default(const pr_bcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    PAGERANK_TIME_START(INIT)

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    pr_float *restrict *restrict src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict dst = memory_talloc(pr_float*, bcount);

    const pr_float init = 1.0 / vcount;

    for (graph_size_t block = 0; block < bcount; block++)
    {
        const graph_size_t block_len = graph->blocks_diag[block]->vcount;
              pr_float    *block_src = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
              pr_float    *block_dst = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));

        cpu_pagerank_fill_arr(block_dst, init, block_len);

        src[block] = block_src;
        dst[block] = block_dst;
    }
    PAGERANK_TIME_STOP(INIT)

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            SWAP_VALUES(src, dst);

            PAGERANK_TIME_START(BASERANK)
            pr_float base_rank = 0.0;

            for (graph_size_t block = 0; block < bcount; block++)
                base_rank += cpu_pagerank_baserank(
                    src[block],
                    graph->blocks_diag[block]->deg_i,
                    graph->blocks_diag[block]->vcount
                );

            base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);
            PAGERANK_TIME_STOP(BASERANK)

            PAGERANK_TIME_START(UPDATE)
            for (graph_size_t block = 0; block < bcount; block++)
                cpu_pagerank_fill_arr(dst[block], 0.0, graph->blocks_diag[block]->vcount);

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                    cpu_pagerank_update_rank_pull(
                        dst[block_row],
                        src[block_col],
                        graph->blocks[block_row*bcount+block_col]->row_idx,
                        graph->blocks[block_row*bcount+block_col]->col_idx,
                        graph->blocks[block_row*bcount+block_col]->deg_i,
                        graph->blocks[block_row*bcount+block_col]->vcount
                    );

            for (graph_size_t block = 0; block < bcount; block++)
                cpu_pagerank_update_dest(
                    dst[block],
                    base_rank,
                    options->damping,
                    graph->blocks_diag[block]->vcount
                );
            PAGERANK_TIME_STOP(UPDATE)
        }

        PAGERANK_TIME_START(DIFF)
        diff = 0.0;

        for (graph_size_t block = 0; block < bcount; block++)
            diff += cpu_pagerank_calc_diff(
                src[block],
                dst[block],
                graph->blocks_diag[block]->vcount
            );
        PAGERANK_TIME_STOP(DIFF)

        iterations += options->local_iterations;
    }

    if (options->result != NULL)
    {
        PAGERANK_TIME_START(TRANSFER)
        for (graph_size_t block = 0; block < bcount; block++)
            cpu_pagerank_read_col(
                options->result,
                dst[block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->blocks_diag[block]->vcount
            );
        PAGERANK_TIME_STOP(TRANSFER)
    }

    for (graph_size_t block = 0; block < bcount; block++)
    {
        memory_free((void*)src[block]);
        memory_free((void*)dst[block]);
    }

    memory_free((void*)src);
    memory_free((void*)dst);

    return iterations;
}

uint32_t pagerank_bcsc_ref_stepped(const pr_bcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    PAGERANK_TIME_START(INIT)

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    pr_float *restrict *restrict src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict dst = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict tmp = memory_talloc(pr_float*, bcount);

    const pr_float init = 1.0 / vcount;

    for (graph_size_t block = 0; block < bcount; block++)
    {
        const graph_size_t block_len = graph->blocks_diag[block]->vcount;
              pr_float    *block_src = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
              pr_float    *block_dst = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
              pr_float    *block_tmp = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));

        cpu_pagerank_fill_arr(block_dst, init, block_len);

        src[block] = block_src;
        dst[block] = block_dst;
        tmp[block] = block_tmp;
    }
    PAGERANK_TIME_STOP(INIT)

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            SWAP_VALUES(src, dst);

            PAGERANK_TIME_START(BASERANK)
            pr_float base_rank = 0.0;

            for (graph_size_t block = 0; block < bcount; block++)
                base_rank += cpu_pagerank_baserank(
                    src[block],
                    graph->blocks_diag[block]->deg_i,
                    graph->blocks_diag[block]->vcount
                );

            base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);
            PAGERANK_TIME_STOP(BASERANK)

            PAGERANK_TIME_START(UPDATE)
            for (graph_size_t block = 0; block < bcount; block++)
                cpu_pagerank_update_tmp(
                    tmp[block],
                    src[block],
                    graph->blocks_diag[block]->deg_i,
                    graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                cpu_pagerank_fill_arr(dst[block], 0.0, graph->blocks_diag[block]->vcount);

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                    cpu_pagerank_update_rank_tmp_pull(
                        dst[block_row],
                        tmp[block_col],
                        graph->blocks[block_row*bcount+block_col]->row_idx,
                        graph->blocks[block_row*bcount+block_col]->col_idx,
                        graph->blocks[block_row*bcount+block_col]->vcount
                    );

            for (graph_size_t block = 0; block < bcount; block++)
                cpu_pagerank_update_dest(
                    dst[block],
                    base_rank,
                    options->damping,
                    graph->blocks_diag[block]->vcount
                );
            PAGERANK_TIME_STOP(UPDATE)
        }

        PAGERANK_TIME_START(DIFF)
        diff = 0.0;

        for (graph_size_t block = 0; block < bcount; block++)
            diff += cpu_pagerank_calc_diff(
                src[block],
                dst[block],
                graph->blocks_diag[block]->vcount
            );
        PAGERANK_TIME_STOP(DIFF)

        iterations += options->local_iterations;
    }

    if (options->result != NULL)
    {
        PAGERANK_TIME_START(TRANSFER)
        for (graph_size_t block = 0; block < bcount; block++)
            cpu_pagerank_read_col(
                options->result,
                dst[block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->blocks_diag[block]->vcount
            );
        PAGERANK_TIME_STOP(TRANSFER)
    }

    for (graph_size_t block = 0; block < bcount; block++)
    {
        memory_free((void*)src[block]);
        memory_free((void*)dst[block]);
        memory_free((void*)tmp[block]);
    }

    memory_free((void*)src);
    memory_free((void*)dst);
    memory_free((void*)tmp);

    return iterations;
}

uint32_t pagerank_bcsc_ref_mapped(const pr_bcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    PAGERANK_TIME_START(INIT)

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    pr_float *restrict *restrict src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict dst = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict tmp = memory_talloc(pr_float*, bcount);

    const pr_float init = 1.0 / vcount;

    for (graph_size_t block = 0; block < bcount; block++)
    {
        const graph_size_t block_len = graph->blocks_diag[block]->vcount;
              pr_float    *block_src = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
              pr_float    *block_dst = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
              pr_float    *block_tmp = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));

        cpu_pagerank_fill_arr(block_dst, init, block_len);

        src[block] = block_src;
        dst[block] = block_dst;
        tmp[block] = block_tmp;
    }
    PAGERANK_TIME_STOP(INIT)

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            SWAP_VALUES(src, dst);

            PAGERANK_TIME_START(BASERANK)
            pr_float base_rank = 0.0;

            for (graph_size_t block = 0; block < bcount; block++)
                cpu_pagerank_baserank_mapped(
                    tmp[block],
                    src[block],
                    graph->blocks_diag[block]->deg_i,
                    graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                base_rank += cpu_pagerank_sum_arr(tmp[block], graph->blocks_diag[block]->vcount);

            base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);
            PAGERANK_TIME_STOP(BASERANK)

            PAGERANK_TIME_START(UPDATE)
            for (graph_size_t block = 0; block < bcount; block++)
                cpu_pagerank_update_tmp(
                    tmp[block],
                    src[block],
                    graph->blocks_diag[block]->deg_i,
                    graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                cpu_pagerank_fill_arr(dst[block], 0.0, graph->blocks_diag[block]->vcount);

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                    cpu_pagerank_update_rank_tmp_pull(
                        dst[block_row],
                        tmp[block_col],
                        graph->blocks[block_row*bcount+block_col]->row_idx,
                        graph->blocks[block_row*bcount+block_col]->col_idx,
                        graph->blocks[block_row*bcount+block_col]->vcount
                    );

            for (graph_size_t block = 0; block < bcount; block++)
                cpu_pagerank_update_dest(
                    dst[block],
                    base_rank,
                    options->damping,
                    graph->blocks_diag[block]->vcount
                );
            PAGERANK_TIME_STOP(UPDATE)
        }

        PAGERANK_TIME_START(DIFF)
        for (graph_size_t block = 0; block < bcount; block++)
            cpu_pagerank_calc_diff_mapped(
                tmp[block],
                src[block],
                dst[block],
                graph->blocks_diag[block]->vcount
            );

        diff = 0.0;
        for (graph_size_t block = 0; block < bcount; block++)
            diff += cpu_pagerank_sum_arr(tmp[block], graph->blocks_diag[block]->vcount);
        PAGERANK_TIME_STOP(DIFF)

        iterations += options->local_iterations;
    }

    if (options->result != NULL)
    {
        PAGERANK_TIME_START(TRANSFER)
        for (graph_size_t block = 0; block < bcount; block++)
            cpu_pagerank_read_col(
                options->result,
                dst[block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->blocks_diag[block]->vcount
            );
        PAGERANK_TIME_STOP(TRANSFER)
    }

    for (graph_size_t block = 0; block < bcount; block++)
    {
        memory_free((void*)src[block]);
        memory_free((void*)dst[block]);
        memory_free((void*)tmp[block]);
    }

    memory_free((void*)src);
    memory_free((void*)dst);
    memory_free((void*)tmp);

    return iterations;
}
