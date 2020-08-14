// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/cpu_codelets.h"
#include "alg/pagerank/omp_codelets.h"
#include "graph/pcsr_mpi.h"
#include "util/math.h"
#include "util/memory.h"
#include "util/mpi.h"

uint32_t pagerank_pcsc_mpi_default(MPI_Comm comm, const pr_pcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    assert(comm != MPI_COMM_NULL);
    PAGERANK_TIME_START(INIT)

    const MPI_Datatype MPI_PR_FLOAT = mpi_get_float_type(pr_float);

    const graph_size_t vcount = graph->vcount;
    const graph_size_t bcount = graph->local_graph->bcount;

    pr_float *restrict *restrict src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict dst = memory_talloc(pr_float*, bcount);

    pr_float *restrict sbuf = memory_talloc(pr_float, pcsr_cross_graph(_get_vertex_degree_out)(graph->cross_graph, graph->part));
    pr_float *restrict rbuf = memory_talloc(pr_float, pcsr_cross_graph(_get_vertex_degree_in)(graph->cross_graph, graph->part));

    const pr_float init = 1.0 / vcount;

    for (graph_size_t block = 0; block < bcount; block++)
    {
        const graph_size_t block_len = graph->local_graph->blocks_diag[block]->vcount;
                 pr_float *block_src = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
                 pr_float *block_dst = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));

        omp_pagerank_fill_arr(block_dst, init, block_len);

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
                base_rank += omp_pagerank_baserank(
                    src[block],
                    graph->local_graph->blocks_diag[block]->deg_i,
                    graph->local_graph->blocks_diag[block]->vcount
                );

            MPI_Allreduce(MPI_IN_PLACE, &base_rank, 1, MPI_PR_FLOAT, MPI_SUM, comm);

            base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);
            PAGERANK_TIME_STOP(BASERANK)

            PAGERANK_TIME_START(UPDATE)
            for (graph_size_t block = 0; block < bcount; block++)
                omp_pagerank_fill_arr(dst[block], 0.0, graph->local_graph->blocks_diag[block]->vcount);

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                    omp_pagerank_update_rank_pull(
                        dst[block_row],
                        src[block_col],
                        graph->local_graph->blocks[block_row*bcount+block_col]->row_idx,
                        graph->local_graph->blocks[block_row*bcount+block_col]->col_idx,
                        graph->local_graph->blocks[block_row*bcount+block_col]->deg_i,
                        graph->local_graph->blocks[block_row*bcount+block_col]->vcount
                    );
            PAGERANK_TIME_STOP(UPDATE)

            PAGERANK_TIME_START(TRANSFER)
            pcsr_graph_sync_fan_in_ex(graph, pr_pcsr_local_graph(_get_vertex_index), dst, MPI_PR_FLOAT, rbuf, sbuf)
            PAGERANK_TIME_STOP(TRANSFER)

            PAGERANK_TIME_MARK(UPDATE)
            for (graph_size_t block = 0; block < bcount; block++)
                omp_pagerank_update_dest(
                    dst[block],
                    base_rank,
                    options->damping,
                    graph->local_graph->blocks_diag[block]->vcount
                );
            PAGERANK_TIME_STOP(UPDATE)

            PAGERANK_TIME_MARK(TRANSFER)
            pcsr_graph_sync_fan_out_ex(graph, pr_pcsr_local_graph(_get_vertex_index), dst, MPI_PR_FLOAT, sbuf, rbuf)
            PAGERANK_TIME_STOP(TRANSFER)
        }

        PAGERANK_TIME_START(DIFF)
        diff = 0.0;

        for (graph_size_t block = 0; block < bcount; block++)
            diff += omp_pagerank_calc_diff(
                src[block],
                dst[block],
                graph->local_graph->blocks_diag[block]->vcount
            );

        MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_PR_FLOAT, MPI_SUM, comm);
        PAGERANK_TIME_STOP(DIFF)

        iterations += options->local_iterations;
    }

    PAGERANK_TIME_START(TRANSFER)
    char write_res = (char) (options->result != NULL);
    MPI_Bcast(&write_res, 1, MPI_CHAR, mpi_get_root(), comm);

    if (write_res)
    {
        const graph_size_t pcount = graph->pcount;
        int sendcount = (int)(graph->local_graph->vcount - pcsr_cross_graph(_get_vertex_degree_in)(graph->cross_graph, graph->part));

        pr_float *restrict tmp_res = memory_talloc(pr_float, graph->local_graph->vcount + graph->local_graph->bcount*BCSR_GRAPH_VERTEX_PACK);
        for (graph_size_t block = 0; block < bcount; block++)
            omp_pagerank_read_col(
                tmp_res,
                dst[block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->local_graph->blocks_diag[block]->vcount
            );

        int recvcounts[pcount];
        MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, mpi_get_root(), comm);

        int displs[pcount]; displs[0] = 0;
        for (graph_size_t p = 1; p < pcount; p++)
            displs[p] = displs[p-1] + recvcounts[p-1];
        MPI_Gatherv(tmp_res, sendcount, MPI_PR_FLOAT, options->result, recvcounts, displs, MPI_PR_FLOAT, mpi_get_root(), comm);

        memory_free((void*)tmp_res);
    }
    PAGERANK_TIME_STOP(TRANSFER)

    for (graph_size_t block = 0; block < bcount; block++)
    {
        memory_free((void*)src[block]);
        memory_free((void*)dst[block]);
    }

    memory_free((void*)sbuf);
    memory_free((void*)rbuf);

    memory_free((void*)src);
    memory_free((void*)dst);

    return iterations;
}

uint32_t pagerank_pcsc_mpi_stepped(MPI_Comm comm, const pr_pcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    assert(comm != MPI_COMM_NULL);
    PAGERANK_TIME_START(INIT)

    const MPI_Datatype MPI_PR_FLOAT = mpi_get_float_type(pr_float);

    const graph_size_t vcount = graph->vcount;
    const graph_size_t bcount = graph->local_graph->bcount;

    pr_float *restrict *restrict src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict dst = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict tmp = memory_talloc(pr_float*, bcount);

    pr_float *restrict sbuf = memory_talloc(pr_float, pcsr_cross_graph(_get_vertex_degree_out)(graph->cross_graph, graph->part));
    pr_float *restrict rbuf = memory_talloc(pr_float, pcsr_cross_graph(_get_vertex_degree_in)(graph->cross_graph, graph->part));

    const pr_float init = 1.0 / vcount;

    for (graph_size_t block = 0; block < bcount; block++)
    {
        const graph_size_t  block_len = graph->local_graph->blocks_diag[block]->vcount;
              pr_float     *block_src = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
              pr_float     *block_dst = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
              pr_float     *block_tmp = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));

        omp_pagerank_fill_arr(block_dst, init, block_len);

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
                base_rank += omp_pagerank_baserank(
                    src[block],
                    graph->local_graph->blocks_diag[block]->deg_i,
                    graph->local_graph->blocks_diag[block]->vcount
                );

            MPI_Allreduce(MPI_IN_PLACE, &base_rank, 1, MPI_PR_FLOAT, MPI_SUM, comm);

            base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);
            PAGERANK_TIME_STOP(BASERANK)

            PAGERANK_TIME_START(UPDATE)
            for (graph_size_t block = 0; block < bcount; block++)
                omp_pagerank_update_tmp(
                    tmp[block],
                    src[block],
                    graph->local_graph->blocks_diag[block]->deg_i,
                    graph->local_graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                omp_pagerank_fill_arr(dst[block], 0.0, graph->local_graph->blocks_diag[block]->vcount);

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                    omp_pagerank_update_rank_tmp_pull_binsearch(
                        dst[block_row],
                        tmp[block_col],
                        graph->local_graph->blocks[block_row*bcount+block_col]->row_idx,
                        graph->local_graph->blocks[block_row*bcount+block_col]->col_idx,
                        graph->local_graph->blocks[block_row*bcount+block_col]->vcount
                    );
            PAGERANK_TIME_STOP(UPDATE)

            PAGERANK_TIME_START(TRANSFER)
            pcsr_graph_sync_fan_in_ex(graph, pr_pcsr_local_graph(_get_vertex_index), dst, MPI_PR_FLOAT, rbuf, sbuf)
            PAGERANK_TIME_STOP(TRANSFER)

            PAGERANK_TIME_MARK(UPDATE)
            for (graph_size_t block = 0; block < bcount; block++)
                omp_pagerank_update_dest(
                    dst[block],
                    base_rank,
                    options->damping,
                    graph->local_graph->blocks_diag[block]->vcount
                );
            PAGERANK_TIME_STOP(UPDATE)

            PAGERANK_TIME_MARK(TRANSFER)
            pcsr_graph_sync_fan_out_ex(graph, pr_pcsr_local_graph(_get_vertex_index), dst, MPI_PR_FLOAT, sbuf, rbuf)
            PAGERANK_TIME_STOP(TRANSFER)
        }

        PAGERANK_TIME_START(DIFF)
        diff = 0.0;

        for (graph_size_t block = 0; block < bcount; block++)
            diff += omp_pagerank_calc_diff(
                src[block],
                dst[block],
                graph->local_graph->blocks_diag[block]->vcount
            );

        MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_PR_FLOAT, MPI_SUM, comm);
        PAGERANK_TIME_STOP(DIFF)

        iterations += options->local_iterations;
    }

    PAGERANK_TIME_START(TRANSFER)
    char write_res = (char) (options->result != NULL);
    MPI_Bcast(&write_res, 1, MPI_CHAR, mpi_get_root(), comm);

    if (write_res)
    {
        const graph_size_t pcount = graph->pcount;
        int sendcount = (int)(graph->local_graph->vcount - pcsr_cross_graph(_get_vertex_degree_in)(graph->cross_graph, graph->part));

        pr_float *restrict tmp_res = memory_talloc(pr_float, graph->local_graph->vcount + graph->local_graph->bcount*BCSR_GRAPH_VERTEX_PACK);
        for (graph_size_t block = 0; block < bcount; block++)
            omp_pagerank_read_col(
                tmp_res,
                dst[block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->local_graph->blocks_diag[block]->vcount
            );

        int recvcounts[pcount];
        MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, mpi_get_root(), comm);

        int displs[pcount]; displs[0] = 0;
        for (graph_size_t p = 1; p < pcount; p++)
            displs[p] = displs[p-1] + recvcounts[p-1];
        MPI_Gatherv(tmp_res, sendcount, MPI_PR_FLOAT, options->result, recvcounts, displs, MPI_PR_FLOAT, mpi_get_root(), comm);

        memory_free((void*)tmp_res);
    }
    PAGERANK_TIME_STOP(TRANSFER)

    for (graph_size_t block = 0; block < bcount; block++)
    {
        memory_free((void*)src[block]);
        memory_free((void*)dst[block]);
        memory_free((void*)tmp[block]);
    }

    memory_free((void*)sbuf);
    memory_free((void*)rbuf);

    memory_free((void*)src);
    memory_free((void*)dst);
    memory_free((void*)tmp);

    return iterations;
}

uint32_t pagerank_pcsc_mpi_mapped(MPI_Comm comm, const pr_pcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    assert(comm != MPI_COMM_NULL);
    PAGERANK_TIME_START(INIT)

    const MPI_Datatype MPI_PR_FLOAT = mpi_get_float_type(pr_float);

    const graph_size_t vcount = graph->vcount;
    const graph_size_t bcount = graph->local_graph->bcount;

    pr_float *restrict *restrict src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict dst = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict tmp = memory_talloc(pr_float*, bcount);

    pr_float *restrict sbuf = memory_talloc(pr_float, pcsr_cross_graph(_get_vertex_degree_out)(graph->cross_graph, graph->part));
    pr_float *restrict rbuf = memory_talloc(pr_float, pcsr_cross_graph(_get_vertex_degree_in)(graph->cross_graph, graph->part));

    const pr_float init = 1.0 / vcount;

    for (graph_size_t block = 0; block < bcount; block++)
    {
        const graph_size_t  block_len = graph->local_graph->blocks_diag[block]->vcount;
              pr_float     *block_src = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
              pr_float     *block_dst = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
              pr_float     *block_tmp = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));

        omp_pagerank_fill_arr(block_dst, init, block_len);

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
                omp_pagerank_baserank_mapped(
                    tmp[block],
                    src[block],
                    graph->local_graph->blocks_diag[block]->deg_i,
                    graph->local_graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                base_rank += omp_pagerank_sum_arr(tmp[block], graph->local_graph->blocks_diag[block]->vcount);

            MPI_Allreduce(MPI_IN_PLACE, &base_rank, 1, MPI_PR_FLOAT, MPI_SUM, comm);

            base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);
            PAGERANK_TIME_STOP(BASERANK)

            PAGERANK_TIME_START(UPDATE)
            for (graph_size_t block = 0; block < bcount; block++)
                omp_pagerank_update_tmp(
                    tmp[block],
                    src[block],
                    graph->local_graph->blocks_diag[block]->deg_i,
                    graph->local_graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                omp_pagerank_fill_arr(dst[block], 0.0, graph->local_graph->blocks_diag[block]->vcount);

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                    omp_pagerank_update_rank_tmp_pull_binsearch(
                        dst[block_row],
                        tmp[block_col],
                        graph->local_graph->blocks[block_row*bcount+block_col]->row_idx,
                        graph->local_graph->blocks[block_row*bcount+block_col]->col_idx,
                        graph->local_graph->blocks[block_row*bcount+block_col]->vcount
                    );
            PAGERANK_TIME_STOP(UPDATE)

            PAGERANK_TIME_START(TRANSFER)
            pcsr_graph_sync_fan_in_ex(graph, pr_pcsr_local_graph(_get_vertex_index), dst, MPI_PR_FLOAT, rbuf, sbuf)
            PAGERANK_TIME_STOP(TRANSFER)

            PAGERANK_TIME_MARK(UPDATE)
            for (graph_size_t block = 0; block < bcount; block++)
                omp_pagerank_update_dest(
                    dst[block],
                    base_rank,
                    options->damping,
                    graph->local_graph->blocks_diag[block]->vcount
                );
            PAGERANK_TIME_STOP(UPDATE)

            PAGERANK_TIME_MARK(TRANSFER)
            pcsr_graph_sync_fan_out_ex(graph, pr_pcsr_local_graph(_get_vertex_index), dst, MPI_PR_FLOAT, sbuf, rbuf)
            PAGERANK_TIME_STOP(TRANSFER)
        }

        PAGERANK_TIME_START(DIFF)
        diff = 0.0;

        for (graph_size_t block = 0; block < bcount; block++)
            omp_pagerank_calc_diff_mapped(
                tmp[block],
                src[block],
                dst[block],
                graph->local_graph->blocks_diag[block]->vcount
            );

        for (graph_size_t block = 0; block < bcount; block++)
            diff += omp_pagerank_sum_arr(tmp[block], graph->local_graph->blocks_diag[block]->vcount);

        MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_PR_FLOAT, MPI_SUM, comm);
        PAGERANK_TIME_STOP(DIFF)

        iterations += options->local_iterations;
    }

    PAGERANK_TIME_START(TRANSFER)
    char write_res = (char) (options->result != NULL);
    MPI_Bcast(&write_res, 1, MPI_CHAR, mpi_get_root(), comm);

    if (write_res)
    {
        const graph_size_t pcount = graph->pcount;
        int sendcount = (int)(graph->local_graph->vcount - pcsr_cross_graph(_get_vertex_degree_in)(graph->cross_graph, graph->part));

        pr_float *restrict tmp_res = memory_talloc(pr_float, graph->local_graph->vcount + graph->local_graph->bcount*BCSR_GRAPH_VERTEX_PACK);
        for (graph_size_t block = 0; block < bcount; block++)
            omp_pagerank_read_col(
                tmp_res,
                dst[block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->local_graph->blocks_diag[block]->vcount
            );

        int recvcounts[pcount];
        MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, mpi_get_root(), comm);

        int displs[pcount]; displs[0] = 0;
        for (graph_size_t p = 1; p < pcount; p++)
            displs[p] = displs[p-1] + recvcounts[p-1];
        MPI_Gatherv(tmp_res, sendcount, MPI_PR_FLOAT, options->result, recvcounts, displs, MPI_PR_FLOAT, mpi_get_root(), comm);

        memory_free((void*)tmp_res);
    }
    PAGERANK_TIME_STOP(TRANSFER)

    for (graph_size_t block = 0; block < bcount; block++)
    {
        memory_free((void*)src[block]);
        memory_free((void*)dst[block]);
        memory_free((void*)tmp[block]);
    }

    memory_free((void*)sbuf);
    memory_free((void*)rbuf);

    memory_free((void*)src);
    memory_free((void*)dst);
    memory_free((void*)tmp);

    return iterations;
}
