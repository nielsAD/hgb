// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/omp_codelets.h"
#include "util/memory.h"

uint32_t pagerank_csr_omp_binsearch(const pr_csr_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    PAGERANK_TIME_START(INIT)

    const graph_size_t vcount = graph->vcount;

    pr_float *restrict src = memory_talloc(pr_float, vcount);
    pr_float *restrict dst = memory_talloc(pr_float, vcount);
    pr_float *restrict tmp = memory_talloc(pr_float, vcount);
    PAGERANK_TIME_STOP(INIT)

    const pr_float init = 1.0 / vcount;
    omp_pagerank_fill_arr(dst, init, vcount);

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            SWAP_VALUES(src, dst);

            PAGERANK_TIME_START(BASERANK)
            pr_float base_rank = omp_pagerank_baserank(
                src,
                graph->deg_o,
                vcount
            );
            base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);
            PAGERANK_TIME_STOP(BASERANK)

            PAGERANK_TIME_START(UPDATE)
            omp_pagerank_update_tmp(tmp, src, graph->deg_o, vcount);
            omp_pagerank_fill_arr(dst, 0.0, vcount);
            omp_pagerank_update_rank_tmp_push_binsearch(
                dst,
                tmp,
                graph->row_idx,
                graph->col_idx,
                vcount
            );

            omp_pagerank_update_dest(dst, base_rank, options->damping, vcount);
            PAGERANK_TIME_STOP(UPDATE)
        }

        PAGERANK_TIME_START(DIFF)
        diff = omp_pagerank_calc_diff(
            src,
            dst,
            vcount
        );
        PAGERANK_TIME_STOP(DIFF)

        iterations += options->local_iterations;
    }

    if (options->result != NULL)
    {
        PAGERANK_TIME_START(TRANSFER)
        memcpy(options->result, dst, sizeof(*options->result) * vcount);
        PAGERANK_TIME_STOP(TRANSFER)
    }

    memory_free((void*)src);
    memory_free((void*)dst);
    memory_free((void*)tmp);

    return iterations;
}

uint32_t pagerank_bcsr_omp_default(const pr_bcsr_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    PAGERANK_TIME_START(INIT)

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    const int was_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);

    pr_float *restrict *restrict src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict dst = memory_talloc(pr_float*, bcount);
    PAGERANK_TIME_STOP(INIT)

    uint32_t iterations = 0;
    pr_float base_rank  = 0.0;
    pr_float diff       = 1.0;

    OMP_PAGERANK_TIME_DECLARE()
    OMP_PRAGMA(omp parallel num_threads(bcount))
    {
        OMP_PAGERANK_TIME_START(INIT)

        const graph_size_t block = omp_get_thread_num();
        assert((graph_size_t)omp_get_num_threads() == bcount);
        assert(block < bcount);

        const graph_size_t  block_len = graph->blocks_diag[block]->vcount;
        const graph_size_t *block_deg = graph->blocks_diag[block]->deg_o;

        src[block] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
        dst[block] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));

        const pr_float init = 1.0 / vcount;
        cpu_pagerank_fill_arr(dst[block], init, block_len);

        OMP_PAGERANK_TIME_STOP(INIT)

        uint32_t it = 0;
        while ((it < options->min_iterations) || (it < options->max_iterations && diff > options->epsilon))
        {
            for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
            {
                OMP_PRAGMA(omp barrier)

                OMP_PRAGMA(omp single)
                {
                    SWAP_VALUES(src, dst);

                    diff      = 0.0;
                    base_rank = 0.0;
                }

                OMP_PAGERANK_TIME_START(BASERANK)
                const pr_float base_rank_tmp = cpu_pagerank_baserank(
                    src[block],
                    block_deg,
                    block_len
                );

                OMP_PRAGMA(omp atomic)
                base_rank += base_rank_tmp;
                OMP_PAGERANK_TIME_STOP(BASERANK)

                OMP_PAGERANK_TIME_START(UPDATE)
                cpu_pagerank_fill_arr(dst[block], 0.0, block_len);

                OMP_PRAGMA(omp barrier)

                OMP_PRAGMA(omp single)
                base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);

                for (graph_size_t col = 0; col < bcount; col++)
                    cpu_pagerank_update_rank_push(
                        dst[col],
                        src[block],
                        graph->blocks[block*bcount+col]->row_idx,
                        graph->blocks[block*bcount+col]->col_idx,
                        graph->blocks[block*bcount+col]->deg_o,
                        graph->blocks[block*bcount+col]->vcount
                    );

                OMP_PRAGMA(omp barrier)
                cpu_pagerank_update_dest(dst[block], base_rank, options->damping, block_len);
                OMP_PAGERANK_TIME_STOP(UPDATE)
            }

            OMP_PAGERANK_TIME_START(DIFF)
            const pr_float diff_tmp = cpu_pagerank_calc_diff(
                src[block],
                dst[block],
                block_len
            );

            OMP_PRAGMA(omp atomic)
            diff += diff_tmp;
            OMP_PAGERANK_TIME_STOP(DIFF)

            OMP_PRAGMA(omp barrier)
            it += options->local_iterations;
        }

        if (options->result != NULL)
        {
            OMP_PAGERANK_TIME_START(TRANSFER)
            cpu_pagerank_read_col(
                options->result,
                dst[block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->blocks_diag[block]->vcount
            );
            OMP_PAGERANK_TIME_STOP(TRANSFER)
        }

        memory_free((void*)src[block]);
        memory_free((void*)dst[block]);

        OMP_PRAGMA(omp single nowait)
        iterations = it;
    }

    memory_free((void*)src);
    memory_free((void*)dst);

    omp_set_dynamic(was_dynamic);

    return iterations;
}

uint32_t pagerank_bcsr_omp_stepped(const pr_bcsr_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    PAGERANK_TIME_START(INIT)

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    const int was_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);

    pr_float *restrict *restrict src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict dst = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict tmp = memory_talloc(pr_float*, bcount);
    PAGERANK_TIME_STOP(INIT)

    uint32_t iterations = 0;
    pr_float base_rank  = 0.0;
    pr_float diff       = 1.0;

    OMP_PAGERANK_TIME_DECLARE()
    OMP_PRAGMA(omp parallel num_threads(bcount))
    {
        OMP_PAGERANK_TIME_START(INIT)

        const graph_size_t block = omp_get_thread_num();
        assert((graph_size_t)omp_get_num_threads() == bcount);
        assert(block < bcount);

        const graph_size_t  block_len = graph->blocks_diag[block]->vcount;
        const graph_size_t *block_deg = graph->blocks_diag[block]->deg_o;

        src[block] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
        dst[block] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
        tmp[block] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));

        const pr_float init = 1.0 / vcount;
        cpu_pagerank_fill_arr(dst[block], init, block_len);

        OMP_PAGERANK_TIME_STOP(INIT)

        uint32_t it = 0;
        while ((it < options->min_iterations) || (it < options->max_iterations && diff > options->epsilon))
        {
            for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
            {
                OMP_PRAGMA(omp barrier)

                OMP_PRAGMA(omp single)
                {
                    SWAP_VALUES(src, dst);

                    diff      = 0.0;
                    base_rank = 0.0;
                }

                OMP_PAGERANK_TIME_START(BASERANK)
                const pr_float base_rank_tmp = cpu_pagerank_baserank(
                    src[block],
                    block_deg,
                    block_len
                );

                OMP_PRAGMA(omp atomic)
                base_rank += base_rank_tmp;
                OMP_PAGERANK_TIME_STOP(BASERANK)

                OMP_PAGERANK_TIME_START(UPDATE)
                cpu_pagerank_fill_arr(dst[block], 0.0, block_len);

                OMP_PRAGMA(omp barrier)

                cpu_pagerank_update_tmp(tmp[block], src[block], block_deg, block_len);

                OMP_PRAGMA(omp single)
                base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);

                for (graph_size_t col = 0; col < bcount; col++)
                    cpu_pagerank_update_rank_tmp_push(
                        dst[col],
                        tmp[block],
                        graph->blocks[block*bcount+col]->row_idx,
                        graph->blocks[block*bcount+col]->col_idx,
                        graph->blocks[block*bcount+col]->vcount
                    );

                OMP_PRAGMA(omp barrier)

                cpu_pagerank_update_dest(dst[block], base_rank, options->damping, block_len);
                OMP_PAGERANK_TIME_STOP(UPDATE)
            }

            OMP_PAGERANK_TIME_START(DIFF)
            const pr_float diff_tmp = cpu_pagerank_calc_diff(
                src[block],
                dst[block],
                block_len
            );

            OMP_PRAGMA(omp atomic)
            diff += diff_tmp;
            OMP_PAGERANK_TIME_STOP(DIFF)

            OMP_PRAGMA(omp barrier)
            it += options->local_iterations;
        }

        if (options->result != NULL)
        {
            OMP_PAGERANK_TIME_START(TRANSFER)
            cpu_pagerank_read_col(
                options->result,
                dst[block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->blocks_diag[block]->vcount
            );
            OMP_PAGERANK_TIME_STOP(TRANSFER)
        }

        memory_free((void*)src[block]);
        memory_free((void*)dst[block]);
        memory_free((void*)tmp[block]);

        OMP_PRAGMA(omp single nowait)
        iterations = it;
    }

    memory_free((void*)src);
    memory_free((void*)dst);
    memory_free((void*)tmp);

    omp_set_dynamic(was_dynamic);

    return iterations;
}

uint32_t pagerank_bcsr_omp_mapped(const pr_bcsr_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    PAGERANK_TIME_START(INIT)

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    const int was_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);

    pr_float *restrict *restrict src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict dst = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict tmp = memory_talloc(pr_float*, bcount);
    PAGERANK_TIME_STOP(INIT)

    uint32_t  iterations = 0;
    pr_float  base_rank  = 0.0;
    pr_float  diff       = 1.0;

    OMP_PAGERANK_TIME_DECLARE()
    OMP_PRAGMA(omp parallel num_threads(bcount))
    {
        OMP_PAGERANK_TIME_START(INIT)

        const graph_size_t block = omp_get_thread_num();
        assert((graph_size_t)omp_get_num_threads() == bcount);
        assert(block < bcount);

        const graph_size_t  block_len = graph->blocks_diag[block]->vcount;
        const graph_size_t *block_deg = graph->blocks_diag[block]->deg_o;

        src[block] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
        dst[block] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
        tmp[block] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));

        const pr_float init = 1.0 / vcount;
        cpu_pagerank_fill_arr(dst[block], init, block_len);

        OMP_PAGERANK_TIME_STOP(INIT)

        uint32_t it = 0;
        while ((it < options->min_iterations) || (it < options->max_iterations && diff > options->epsilon))
        {
            for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
            {
                OMP_PRAGMA(omp barrier)

                OMP_PRAGMA(omp single)
                {
                    SWAP_VALUES(src, dst);

                    diff      = 0.0;
                    base_rank = 0.0;
                }

                OMP_PAGERANK_TIME_START(BASERANK)
                cpu_pagerank_baserank_mapped(tmp[block], src[block], block_deg, block_len);

                const pr_float base_rank_tmp = cpu_pagerank_sum_arr(
                    tmp[block],
                    block_len
                );

                OMP_PRAGMA(omp atomic)
                base_rank += base_rank_tmp;
                OMP_PAGERANK_TIME_STOP(BASERANK)

                OMP_PAGERANK_TIME_START(UPDATE)
                cpu_pagerank_fill_arr(dst[block], 0.0, block_len);

                OMP_PRAGMA(omp barrier)

                cpu_pagerank_update_tmp(tmp[block], src[block], block_deg, block_len);

                OMP_PRAGMA(omp single)
                base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);

                for (graph_size_t col = 0; col < bcount; col++)
                    cpu_pagerank_update_rank_tmp_push(
                        dst[col],
                        tmp[block],
                        graph->blocks[block*bcount+col]->row_idx,
                        graph->blocks[block*bcount+col]->col_idx,
                        graph->blocks[block*bcount+col]->vcount
                    );

                OMP_PRAGMA(omp barrier)

                cpu_pagerank_update_dest(dst[block], base_rank, options->damping, block_len);
                OMP_PAGERANK_TIME_STOP(UPDATE)
            }

            OMP_PRAGMA(omp barrier)

            OMP_PAGERANK_TIME_START(DIFF)
            cpu_pagerank_calc_diff_mapped(tmp[block], src[block], dst[block], block_len);

            const pr_float diff_tmp = cpu_pagerank_sum_arr(
                tmp[block],
                block_len
            );

            OMP_PRAGMA(omp atomic)
            diff += diff_tmp;
            OMP_PAGERANK_TIME_STOP(DIFF)

            OMP_PRAGMA(omp barrier)
            it += options->local_iterations;
        }

        if (options->result != NULL)
        {
            OMP_PAGERANK_TIME_START(TRANSFER)
            cpu_pagerank_read_col(
                options->result,
                dst[block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->blocks_diag[block]->vcount
            );
            OMP_PAGERANK_TIME_STOP(TRANSFER)
        }

        memory_free((void*)src[block]);
        memory_free((void*)dst[block]);
        memory_free((void*)tmp[block]);

        OMP_PRAGMA(omp single nowait)
        iterations = it;
    }

    memory_free((void*)src);
    memory_free((void*)dst);
    memory_free((void*)tmp);

    omp_set_dynamic(was_dynamic);

    return iterations;
}

uint32_t pagerank_bcsr_omp_redux(const pr_bcsr_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    PAGERANK_TIME_START(INIT)

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    const int was_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);

    pr_float *restrict *restrict src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict dst = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict tmp = memory_talloc(pr_float*, bcount);

    pr_float *restrict *restrict tmp_dst = memory_talloc(pr_float*, bcount * bcount);
    PAGERANK_TIME_STOP(INIT)

    uint32_t  iterations = 0;
    pr_float  base_rank  = 0.0;
    pr_float  diff       = 1.0;

    OMP_PAGERANK_TIME_DECLARE()
    OMP_PRAGMA(omp parallel num_threads(bcount*bcount))
    {
        OMP_PAGERANK_TIME_START(INIT)

        const graph_size_t block = omp_get_thread_num();
        const graph_size_t row   = block / bcount;
        const graph_size_t col   = block % bcount;

        assert((graph_size_t)omp_get_num_threads() == bcount*bcount);
        assert(block < bcount * bcount);
        assert(row < bcount);
        assert(col < bcount);

        const graph_size_t  block_len = graph->blocks_diag[col]->vcount;
        const graph_size_t *block_deg = graph->blocks_diag[col]->deg_o;

        if (row == col)
        {
            src[row] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
            dst[row] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
            tmp[row] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));

            const pr_float init = 1.0 / vcount;
            cpu_pagerank_fill_arr(dst[row], init, block_len);
        }

        if (row > 0)
            tmp_dst[block] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));

        OMP_PRAGMA(omp barrier)
        OMP_PAGERANK_TIME_STOP(INIT)

        uint32_t it = 0;
        while ((it < options->min_iterations) || (it < options->max_iterations && diff > options->epsilon))
        {
            for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
            {
                OMP_PRAGMA(omp single)
                {
                    SWAP_VALUES(src, dst);

                    diff      = 0.0;
                    base_rank = 0.0;
                }

                if (row == 0)
                    tmp_dst[block] = dst[col];

                if (row == col)
                {
                    OMP_PAGERANK_TIME_START(BASERANK)
                    cpu_pagerank_baserank_mapped(tmp[row], src[row], block_deg, block_len);

                    const pr_float base_rank_tmp = cpu_pagerank_sum_arr(
                        tmp[row],
                        block_len
                    );

                    OMP_PRAGMA(omp atomic)
                    base_rank += base_rank_tmp;
                    OMP_PAGERANK_TIME_STOP(BASERANK)
                }

                OMP_PRAGMA(omp barrier)

                OMP_PAGERANK_TIME_START(UPDATE)
                if (row == col)
                    cpu_pagerank_update_tmp(tmp[row], src[row], block_deg, block_len);

                OMP_PRAGMA(omp single)
                base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);

                cpu_pagerank_fill_arr(tmp_dst[block], 0.0, block_len);
                cpu_pagerank_update_rank_tmp_push(
                    tmp_dst[block],
                    tmp[row],
                    graph->blocks[block]->row_idx,
                    graph->blocks[block]->col_idx,
                    graph->blocks[block]->vcount
                );

                OMP_PRAGMA(omp barrier)

                // Manual reduction
                for (graph_size_t rdx = bcount >> 1; rdx > 0; rdx >>= 1)
                {
                    if (row < rdx)
                        cpu_pagerank_add_arr(tmp_dst[block], tmp_dst[block + rdx*bcount], block_len);

                    OMP_PRAGMA(omp barrier)
                }

                if (row == col)
                    cpu_pagerank_update_dest(dst[row], base_rank, options->damping, block_len);

                OMP_PRAGMA(omp barrier)
                OMP_PAGERANK_TIME_STOP(UPDATE)
            }

            if (row == col)
            {
                OMP_PAGERANK_TIME_START(DIFF)
                cpu_pagerank_calc_diff_mapped(tmp[row], src[row], dst[row], block_len);

                const pr_float diff_tmp = cpu_pagerank_sum_arr(
                    tmp[row],
                    block_len
                );

                OMP_PRAGMA(omp atomic)
                diff += diff_tmp;
                OMP_PAGERANK_TIME_STOP(DIFF)
            }

            OMP_PRAGMA(omp barrier)
            it += options->local_iterations;
        }

        if (row == col)
        {
            if (options->result != NULL)
            {
                OMP_PAGERANK_TIME_START(TRANSFER)
                cpu_pagerank_read_col(
                    options->result,
                    dst[row],
                    BCSR_GRAPH_VERTEX_PACK * row,
                    BCSR_GRAPH_VERTEX_PACK * bcount,
                    BCSR_GRAPH_VERTEX_PACK,
                    graph->blocks_diag[row]->vcount
                );
                OMP_PAGERANK_TIME_STOP(TRANSFER)
            }

            memory_free((void*)src[row]);
            memory_free((void*)dst[row]);
            memory_free((void*)tmp[row]);
        }

        if (row > 0)
            memory_free(tmp_dst[block]);

        OMP_PRAGMA(omp single nowait)
        iterations = it;
    }

    memory_free((void*)src);
    memory_free((void*)dst);
    memory_free((void*)tmp);
    memory_free((void*)tmp_dst);

    omp_set_dynamic(was_dynamic);

    return iterations;
}
