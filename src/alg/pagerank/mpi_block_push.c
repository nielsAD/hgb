// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/cpu_codelets.h"
#include "alg/pagerank/omp_codelets.h"
#include "util/math.h"
#include "util/memory.h"
#include "util/mpi.h"

static void mpi_pagerank_read_col(MPI_Comm comm, pr_float *restrict dst, const pr_float *restrict src,  const graph_size_t dst_cols, const graph_size_t src_cols, const graph_size_t size)
{
    const MPI_Datatype dtype = mpi_get_float_type(*dst);

    // http://stackoverflow.com/questions/21831431/mpi-gather-of-columns
    MPI_Datatype recv_type = MPI_DATATYPE_NULL;
    MPI_Type_vector(size / src_cols, src_cols, dst_cols, dtype, &recv_type);
    MPI_Type_commit(&recv_type);

    MPI_Datatype recv_type_resized = MPI_DATATYPE_NULL;
    MPI_Type_create_resized(recv_type, 0, sizeof(*dst) * src_cols, &recv_type_resized);
    MPI_Type_commit(&recv_type_resized);

    // http://stackoverflow.com/questions/30434690/mpi-gather-of-indexed-typed-to-raw-data
    MPI_Datatype send_type = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(size, dtype, &send_type);
    MPI_Type_commit(&send_type);

    MPI_Gather(
        src, 1, send_type,
        dst, 1, recv_type_resized,
        mpi_get_root(), comm
    );

    MPI_Type_free(&send_type);
    MPI_Type_free(&recv_type_resized);
    MPI_Type_free(&recv_type);
}

uint32_t pagerank_bcsr_mpi_default(const pr_bcsr_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    PAGERANK_TIME_START(INIT)

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    assert((graph_size_t)mpi_get_size() >= bcount);
    MPI_Comm comm = mpi_get_sub_comm(MPI_COMM_WORLD, bcount);

    if (comm == MPI_COMM_NULL)
        return 0;

    graph_size_t block;
    {
        int b = 0; MPI_Comm_rank(comm, &b);
        block = b;
        assert(block <= bcount);
    }

    const MPI_Datatype MPI_PR_FLOAT     = mpi_get_float_type(pr_float);
    const MPI_Datatype MPI_GRAPH_SIZE_T = mpi_get_int_type(graph_size_t);

    const graph_size_t  block_len = graph->blocks_diag[block]->vcount;
    const graph_size_t *block_deg = graph->blocks_diag[block]->deg_o;

    graph_size_t max_block_len = 0;
    MPI_Allreduce(&block_len, &max_block_len, 1, MPI_GRAPH_SIZE_T, MPI_MAX, comm);
    max_block_len = ROUND_TO_MULT(max_block_len, BCSR_GRAPH_VERTEX_PACK);

    pr_float *restrict           _src = memory_talloc(pr_float, max_block_len*bcount);
    pr_float *restrict           _dst = memory_talloc(pr_float, max_block_len*bcount);
    pr_float *restrict *restrict  src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict  dst = memory_talloc(pr_float*, bcount);

    for (graph_size_t b = 0; b < bcount; b++)
    {
        src[b] = &_src[b*max_block_len];
        dst[b] = &_dst[b*max_block_len];
    }

    pr_float *restrict block_src = src[block];
    pr_float *restrict block_dst = dst[block];

    const pr_float init = 1.0 / vcount;
    omp_pagerank_fill_arr(block_dst, init, block_len);
    PAGERANK_TIME_STOP(INIT)

    pr_float diff       = 1.0;
    uint32_t iterations = 0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            SWAP_VALUES(src, dst);
            SWAP_VALUES(block_src, block_dst);

            PAGERANK_TIME_START(BASERANK)
            pr_float base_rank = omp_pagerank_baserank(block_src, block_deg, block_len);
            MPI_Allreduce(MPI_IN_PLACE, &base_rank, 1, MPI_PR_FLOAT, MPI_SUM, comm);

            base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);
            PAGERANK_TIME_STOP(BASERANK)

            PAGERANK_TIME_START(UPDATE)
            for (graph_size_t block = 0; block < bcount; block++)
                omp_pagerank_fill_arr(dst[block], 0.0, graph->blocks_diag[block]->vcount);

            for (graph_size_t col = 0; col < bcount; col++)
                omp_pagerank_update_rank_push(
                    dst[col],
                    block_src,
                    graph->blocks[block*bcount+col]->row_idx,
                    graph->blocks[block*bcount+col]->col_idx,
                    graph->blocks[block*bcount+col]->deg_o,
                    graph->blocks[block*bcount+col]->vcount
                );
            PAGERANK_TIME_STOP(UPDATE)

            PAGERANK_TIME_START(TRANSFER)
            for (graph_size_t block = 0; block < bcount; block++)
                MPI_Reduce(MPI_SEND_RECV(dst[block], block_dst), max_block_len, MPI_PR_FLOAT, MPI_SUM, block, comm);
            PAGERANK_TIME_STOP(TRANSFER)

            PAGERANK_TIME_MARK(UPDATE)
            omp_pagerank_update_dest(block_dst, base_rank, options->damping, block_len);
            PAGERANK_TIME_STOP(UPDATE)
        }

        PAGERANK_TIME_START(DIFF)
        diff = omp_pagerank_calc_diff(block_src, block_dst, block_len);
        MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_PR_FLOAT, MPI_SUM, comm);
        PAGERANK_TIME_STOP(DIFF)

        iterations += options->local_iterations;
    }

    PAGERANK_TIME_START(TRANSFER)
    char write_res = (char) (options->result != NULL);
    MPI_Bcast(&write_res, 1, MPI_CHAR, mpi_get_root(), comm);

    if (write_res)
        mpi_pagerank_read_col(
            comm,
            options->result, block_dst,
            BCSR_GRAPH_VERTEX_PACK * bcount,
            BCSR_GRAPH_VERTEX_PACK,
            max_block_len
        );
    PAGERANK_TIME_STOP(TRANSFER)

    memory_free((void*)dst);
    memory_free((void*)src);
    memory_free((void*)_dst);
    memory_free((void*)_src);

    MPI_Comm_free(&comm);

    return iterations;
}

uint32_t pagerank_bcsr_mpi_stepped(const pr_bcsr_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    PAGERANK_TIME_START(INIT)

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    assert((graph_size_t)mpi_get_size() >= bcount);
    MPI_Comm comm = mpi_get_sub_comm(MPI_COMM_WORLD, bcount);

    if (comm == MPI_COMM_NULL)
        return 0;

    graph_size_t block;
    {
        int b = 0; MPI_Comm_rank(comm, &b);
        block = b;
        assert(block <= bcount);
    }

    const MPI_Datatype MPI_PR_FLOAT     = mpi_get_float_type(pr_float);
    const MPI_Datatype MPI_GRAPH_SIZE_T = mpi_get_int_type(graph_size_t);

    const graph_size_t  block_len = graph->blocks_diag[block]->vcount;
    const graph_size_t *block_deg = graph->blocks_diag[block]->deg_o;

    graph_size_t max_block_len = 0;
    MPI_Allreduce(&block_len, &max_block_len, 1, MPI_GRAPH_SIZE_T, MPI_MAX, comm);
    max_block_len = ROUND_TO_MULT(max_block_len, BCSR_GRAPH_VERTEX_PACK);

    pr_float *restrict block_tmp = memory_talloc(pr_float, max_block_len);

    pr_float *restrict           _src = memory_talloc(pr_float, max_block_len*bcount);
    pr_float *restrict           _dst = memory_talloc(pr_float, max_block_len*bcount);
    pr_float *restrict *restrict  src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict  dst = memory_talloc(pr_float*, bcount);

    for (graph_size_t b = 0; b < bcount; b++)
    {
        src[b] = &_src[b*max_block_len];
        dst[b] = &_dst[b*max_block_len];
    }

    pr_float *restrict block_src = src[block];
    pr_float *restrict block_dst = dst[block];

    const pr_float init = 1.0 / vcount;
    omp_pagerank_fill_arr(block_dst, init, block_len);
    PAGERANK_TIME_STOP(INIT)

    pr_float diff       = 1.0;
    uint32_t iterations = 0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            SWAP_VALUES(src, dst);
            SWAP_VALUES(block_src, block_dst);

            PAGERANK_TIME_START(BASERANK)
            pr_float base_rank = omp_pagerank_baserank(block_src, block_deg, block_len);

            omp_pagerank_update_tmp(block_tmp, block_src, block_deg, block_len);
            MPI_Allreduce(MPI_IN_PLACE, &base_rank, 1, MPI_PR_FLOAT, MPI_SUM, comm);

            base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);
            PAGERANK_TIME_STOP(BASERANK)

            PAGERANK_TIME_START(UPDATE)
            for (graph_size_t block = 0; block < bcount; block++)
                omp_pagerank_fill_arr(dst[block], 0.0, block_len);

            for (graph_size_t col = 0; col < bcount; col++)
                omp_pagerank_update_rank_tmp_push(
                    dst[col],
                    block_tmp,
                    graph->blocks[block*bcount+col]->row_idx,
                    graph->blocks[block*bcount+col]->col_idx,
                    graph->blocks[block*bcount+col]->vcount
                );
            PAGERANK_TIME_STOP(UPDATE)

            PAGERANK_TIME_START(TRANSFER)
            for (graph_size_t block = 0; block < bcount; block++)
                MPI_Reduce(MPI_SEND_RECV(dst[block], block_dst), max_block_len, MPI_PR_FLOAT, MPI_SUM, block, comm);
            PAGERANK_TIME_STOP(TRANSFER)

            PAGERANK_TIME_MARK(UPDATE)
            omp_pagerank_update_dest(block_dst, base_rank, options->damping, block_len);
            PAGERANK_TIME_STOP(UPDATE)
        }

        PAGERANK_TIME_START(DIFF)
        diff = omp_pagerank_calc_diff(block_src, block_dst, block_len);
        MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_PR_FLOAT, MPI_SUM, comm);
        PAGERANK_TIME_STOP(DIFF)

        iterations += options->local_iterations;
    }

    PAGERANK_TIME_START(TRANSFER)
    char write_res = (char) (options->result != NULL);
    MPI_Bcast(&write_res, 1, MPI_CHAR, mpi_get_root(), comm);

    if (write_res)
        mpi_pagerank_read_col(
            comm,
            options->result, block_dst,
            BCSR_GRAPH_VERTEX_PACK * bcount,
            BCSR_GRAPH_VERTEX_PACK,
            max_block_len
        );
    PAGERANK_TIME_STOP(TRANSFER)

    memory_free((void*)dst);
    memory_free((void*)src);
    memory_free((void*)_dst);
    memory_free((void*)_src);
    memory_free((void*)block_tmp);

    MPI_Comm_free(&comm);

    return iterations;
}

uint32_t pagerank_bcsr_mpi_mapped(const pr_bcsr_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    PAGERANK_TIME_START(INIT)

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    assert((graph_size_t)mpi_get_size() >= bcount);
    MPI_Comm comm = mpi_get_sub_comm(MPI_COMM_WORLD, bcount);

    if (comm == MPI_COMM_NULL)
        return 0;

    graph_size_t block;
    {
        int b = 0; MPI_Comm_rank(comm, &b);
        block = b;
        assert(block <= bcount);
    }

    const MPI_Datatype MPI_PR_FLOAT     = mpi_get_float_type(pr_float);
    const MPI_Datatype MPI_GRAPH_SIZE_T = mpi_get_int_type(graph_size_t);

    const graph_size_t  block_len = graph->blocks_diag[block]->vcount;
    const graph_size_t *block_deg = graph->blocks_diag[block]->deg_o;

    graph_size_t max_block_len = 0;
    MPI_Allreduce(&block_len, &max_block_len, 1, MPI_GRAPH_SIZE_T, MPI_MAX, comm);
    max_block_len = ROUND_TO_MULT(max_block_len, BCSR_GRAPH_VERTEX_PACK);

    pr_float *restrict block_tmp = memory_talloc(pr_float, max_block_len);

    pr_float *restrict           _src = memory_talloc(pr_float, max_block_len*bcount);
    pr_float *restrict           _dst = memory_talloc(pr_float, max_block_len*bcount);
    pr_float *restrict *restrict  src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict  dst = memory_talloc(pr_float*, bcount);

    for (graph_size_t b = 0; b < bcount; b++)
    {
        src[b] = &_src[b*max_block_len];
        dst[b] = &_dst[b*max_block_len];
    }

    pr_float *restrict block_src = src[block];
    pr_float *restrict block_dst = dst[block];

    const pr_float init = 1.0 / vcount;
    omp_pagerank_fill_arr(block_dst, init, block_len);
    PAGERANK_TIME_STOP(INIT)

    pr_float diff       = 1.0;
    uint32_t iterations = 0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            SWAP_VALUES(src, dst);
            SWAP_VALUES(block_src, block_dst);

            PAGERANK_TIME_START(BASERANK)
            omp_pagerank_baserank_mapped(block_tmp, block_src, block_deg, block_len);
            pr_float base_rank = omp_pagerank_sum_arr(block_tmp, block_len);
            MPI_Allreduce(MPI_IN_PLACE, &base_rank, 1, MPI_PR_FLOAT, MPI_SUM, comm);

            base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);
            PAGERANK_TIME_STOP(BASERANK)

            PAGERANK_TIME_START(UPDATE)
            omp_pagerank_update_tmp(block_tmp, block_src, block_deg, block_len);
            
            for (graph_size_t block = 0; block < bcount; block++)
                omp_pagerank_fill_arr(dst[block], 0.0, block_len);

            for (graph_size_t col = 0; col < bcount; col++)
                omp_pagerank_update_rank_tmp_push(
                    dst[col],
                    block_tmp,
                    graph->blocks[block*bcount+col]->row_idx,
                    graph->blocks[block*bcount+col]->col_idx,
                    graph->blocks[block*bcount+col]->vcount
                );
            PAGERANK_TIME_STOP(UPDATE)

            PAGERANK_TIME_START(TRANSFER)
            for (graph_size_t block = 0; block < bcount; block++)
                MPI_Reduce(MPI_SEND_RECV(dst[block], block_dst), max_block_len, MPI_PR_FLOAT, MPI_SUM, block, comm);
            PAGERANK_TIME_STOP(TRANSFER)

            PAGERANK_TIME_MARK(UPDATE)
            omp_pagerank_update_dest(block_dst, base_rank, options->damping, block_len);
            PAGERANK_TIME_STOP(UPDATE)
        }

        PAGERANK_TIME_START(DIFF)
        omp_pagerank_calc_diff_mapped(block_tmp, block_src, block_dst, block_len);
        diff = omp_pagerank_sum_arr(block_tmp, block_len);
        MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_PR_FLOAT, MPI_SUM, comm);
        PAGERANK_TIME_STOP(DIFF)

        iterations += options->local_iterations;
    }

    PAGERANK_TIME_START(TRANSFER)
    char write_res = (char) (options->result != NULL);
    MPI_Bcast(&write_res, 1, MPI_CHAR, mpi_get_root(), comm);

    if (write_res)
        mpi_pagerank_read_col(
            comm,
            options->result, block_dst,
            BCSR_GRAPH_VERTEX_PACK * bcount,
            BCSR_GRAPH_VERTEX_PACK,
            max_block_len
        );
    PAGERANK_TIME_STOP(TRANSFER)

    memory_free((void*)dst);
    memory_free((void*)src);
    memory_free((void*)_dst);
    memory_free((void*)_src);
    memory_free((void*)block_tmp);

    MPI_Comm_free(&comm);

    return iterations;
}

uint32_t pagerank_bcsr_mpi_redux(const pr_bcsr_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);
    PAGERANK_TIME_START(INIT)

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    assert((graph_size_t)mpi_get_size() >= bcount*bcount);
    MPI_Comm comm = mpi_get_sub_comm(MPI_COMM_WORLD, bcount*bcount);
    if (comm == MPI_COMM_NULL)
        return 0;

    MPI_Comm comm_diag = mpi_get_strided_comm(comm, 0, bcount + 1, bcount*bcount-1);

    MPI_Comm comm_row = MPI_COMM_NULL;
    MPI_Comm comm_col = MPI_COMM_NULL;
    mpi_get_rowcol_comm(comm, bcount, &comm_row, &comm_col);

    assert(comm_row != MPI_COMM_NULL);
    assert(comm_col != MPI_COMM_NULL);

    graph_size_t row, col;
    {
        int r = 0;
        MPI_Comm_rank(comm_row, &r); col = r; assert(col <= bcount);
        MPI_Comm_rank(comm_col, &r); row = r; assert(row <= bcount);
    }

    const MPI_Datatype MPI_PR_FLOAT      = mpi_get_float_type(pr_float);
    const MPI_Datatype MPI_GRAPH_SIZE_T  = mpi_get_int_type(graph_size_t);

    const graph_size_t  block_len = graph->blocks_diag[col]->vcount;
    const graph_size_t *block_deg = graph->blocks_diag[col]->deg_o;

    graph_size_t max_block_len = 0;
    MPI_Allreduce(&block_len, &max_block_len, 1, MPI_GRAPH_SIZE_T, MPI_MAX, comm);
    max_block_len = ROUND_TO_MULT(max_block_len, BCSR_GRAPH_VERTEX_PACK);

    pr_float *restrict block_src = memory_talloc(pr_float, max_block_len);
    pr_float *restrict block_dst = memory_talloc(pr_float, max_block_len);
    pr_float *restrict block_tmp = memory_talloc(pr_float, max_block_len);

    if (row == col)
    {
        const pr_float init = 1.0 / vcount;
        omp_pagerank_fill_arr(block_dst, init, block_len);
    }
    PAGERANK_TIME_STOP(INIT)

    pr_float diff       = 1.0;
    uint32_t iterations = 0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            SWAP_VALUES(block_src, block_dst);

            pr_float base_rank = 0.0;
            if (row == col)
            {
                PAGERANK_TIME_START(BASERANK)
                base_rank = omp_pagerank_baserank(block_src, block_deg, block_len);
                MPI_Allreduce(MPI_IN_PLACE, &base_rank, 1, MPI_PR_FLOAT, MPI_SUM, comm_diag);

                base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);
                PAGERANK_TIME_STOP(BASERANK)

                PAGERANK_TIME_START(UPDATE)
                omp_pagerank_update_tmp(block_tmp, block_src, block_deg, block_len);
                PAGERANK_TIME_STOP(UPDATE)
            }

            PAGERANK_TIME_START(UPDATE)
            MPI_Bcast(block_tmp, max_block_len, MPI_PR_FLOAT, row, comm_row);
            omp_pagerank_fill_arr(block_dst, 0.0, block_len);

            omp_pagerank_update_rank_tmp_push(
                block_dst,
                block_tmp,
                graph->blocks[row*bcount+col]->row_idx,
                graph->blocks[row*bcount+col]->col_idx,
                graph->blocks[row*bcount+col]->vcount
            );
            PAGERANK_TIME_STOP(UPDATE)

            PAGERANK_TIME_START(TRANSFER)
            MPI_Reduce(MPI_SEND_RECV(block_dst, block_tmp), block_len, MPI_PR_FLOAT, MPI_SUM, col, comm_col);
            PAGERANK_TIME_STOP(TRANSFER)

            if (row == col)
            {
                PAGERANK_TIME_START(UPDATE)
                omp_pagerank_calc_dest(block_dst, block_tmp, base_rank, options->damping, block_len);
                PAGERANK_TIME_STOP(UPDATE)
            }
        }

        PAGERANK_TIME_START(DIFF)
        diff = (row == col)
            ? omp_pagerank_calc_diff(block_src, block_dst, block_len)
            : 0.0;
        MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_PR_FLOAT, MPI_SUM, comm);
        PAGERANK_TIME_STOP(DIFF)

        iterations += options->local_iterations;
    }

    if (row == col)
    {
        PAGERANK_TIME_START(TRANSFER)
        char write_res = (char) (options->result != NULL);
        MPI_Bcast(&write_res, 1, MPI_CHAR, mpi_get_root(), comm_diag);

        if (write_res)
            mpi_pagerank_read_col(
                comm_diag,
                options->result, block_dst,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                max_block_len
            );
        PAGERANK_TIME_STOP(TRANSFER)

        MPI_Comm_free(&comm_diag);
    }

    MPI_Comm_free(&comm_col);
    MPI_Comm_free(&comm_row);
    MPI_Comm_free(&comm);

    memory_free((void*)block_tmp);
    memory_free((void*)block_dst);
    memory_free((void*)block_src);

    return iterations;
}
