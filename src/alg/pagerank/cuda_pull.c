// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/cuda_codelets.h"
#include "alg/pagerank/cpu_codelets.h"
#include "alg/pagerank/cuda_problem.h"
#include "util/math.h"
#include "util/memory.h"

static inline void cuda_pr_problem_swap_src_dst(cuda_pr_problem_t *problem)
{
    const graph_size_t bcount = problem->graph->bcount;

    for (graph_size_t b = 0; b < bcount*bcount; b++)
        SWAP_VALUES(problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][b], problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][b]);
}

uint32_t pagerank_csc_cud_lib(const pr_csc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);

    const graph_size_t vcount = graph->vcount;
    const graph_size_t ecount = graph->ecount;

    cusparseMatDescr_t mat_descr; CUDA_ASSERT(cusparseCreateMatDescr(&mat_descr));

    pr_float *restrict src; CUDA_ASSERT(cudaMalloc((void**) &src, sizeof(pr_float) * vcount));
    pr_float *restrict dst; CUDA_ASSERT(cudaMalloc((void**) &dst, sizeof(pr_float) * vcount));
    pr_float *restrict tmp; CUDA_ASSERT(cudaMalloc((void**) &tmp, sizeof(pr_float) * vcount));
    pr_float *restrict val; CUDA_ASSERT(cudaMalloc((void**) &val, sizeof(pr_float) * ecount));

    graph_size_t *restrict degs; CUDA_ASSERT(cudaMalloc((void**) &degs, sizeof(graph_size_t) * vcount));
    graph_size_t *restrict rows; CUDA_ASSERT(cudaMalloc((void**) &rows, sizeof(graph_size_t) * (vcount+1)));
    graph_size_t *restrict cols; CUDA_ASSERT(cudaMalloc((void**) &cols, sizeof(graph_size_t) * ecount));

    CUDA_ASSERT(cudaMemcpyAsync(degs, graph->deg_i,   sizeof(graph_size_t) *  vcount,      cudaMemcpyHostToDevice, NULL));
    CUDA_ASSERT(cudaMemcpyAsync(rows, graph->row_idx, sizeof(graph_size_t) * (vcount + 1), cudaMemcpyHostToDevice, NULL));
    CUDA_ASSERT(cudaMemcpyAsync(cols, graph->col_idx, sizeof(graph_size_t) *  ecount,      cudaMemcpyHostToDevice, NULL));

    cuda_pagerank_fill_arr(NULL, dst, 1.0 / vcount, vcount);
    cuda_pagerank_fill_cols_pull(NULL, val, rows, cols, degs, vcount);

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            SWAP_VALUES(src, dst);

            cuda_pagerank_baserank(NULL, tmp, src, degs, vcount);

            pr_float base_rank;
            CUDA_ASSERT(cublasSasum(handle_cublas, vcount, tmp, 1, &base_rank));

            static const pr_float zero = 0.0;
            static const pr_float one  = 1.0;

            CUDA_ASSERT(cusparseScsrmv(handle_cusparse,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                vcount, vcount, ecount,
                &one, mat_descr,
                val, (int*)rows, (int*)cols,
                src, &zero, dst
            ));

            base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);
            cuda_pagerank_update_dest_raw(NULL, dst, base_rank, options->damping, vcount);
        }

        cuda_pagerank_calc_diff(NULL, tmp, src, dst, vcount);
        CUDA_ASSERT(cublasSasum(handle_cublas, vcount, tmp, 1, &diff));

        iterations += options->local_iterations;
    }

    if (options->result != NULL)
    {
        PAGERANK_TIME_START(TRANSFER)
        CUDA_ASSERT(cudaMemcpy(options->result, dst, sizeof(*options->result) * vcount, cudaMemcpyDeviceToHost));
        PAGERANK_TIME_STOP(TRANSFER)
    }

    CUDA_ASSERT(cudaFree(val));
    CUDA_ASSERT(cudaFree(tmp));
    CUDA_ASSERT(cudaFree(src));
    CUDA_ASSERT(cudaFree(dst));

    CUDA_ASSERT(cudaFree(degs));
    CUDA_ASSERT(cudaFree(rows));
    CUDA_ASSERT(cudaFree(cols));

    CUDA_ASSERT(cusparseDestroyMatDescr(mat_descr));

    return iterations;
}

uint32_t pagerank_bcsc_cuda_mapped(const pr_bcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    PAGERANK_TIME_START(TRANSFER)
    cuda_pr_problem_t *problem = cuda_pr_problem_new_bcsc(graph);
    PAGERANK_TIME_STOP(TRANSFER)

    const pr_float init = 1.0 / vcount;

    for (graph_size_t block = 0; block < bcount; block++)
        cuda_pagerank_fill_arr(
            problem->streams[block],
            problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
            init,
            graph->blocks_diag[block]->vcount
        );

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            cuda_pr_problem_swap_src_dst(problem);

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_baserank(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][block],
                    problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_DEG][block],
                    graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_sum_arr_offset(
                    problem->streams[block],
                    problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                    block,
                    graph->blocks_diag[block]->vcount
                );

            cuda_pr_problem_synchronize(problem);

            cuda_pagerank_sum_arr_offset(
                problem->streams[0],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                0,
                bcount
            );

            cuda_pagerank_baserank_redux(
                problem->streams[0],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                options->damping,
                vcount
            );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_fill_arr(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                    0.0,
                    graph->blocks_diag[block]->vcount
                );

            cuda_pr_problem_synchronize(problem);

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                    cuda_pagerank_update_rank_pull(
                        problem->streams[block_row],
                        problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block_row],
                        problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][block_col],
                        problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_RID][block_row*bcount+block_col],
                        problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_CID][block_row*bcount+block_col],
                        problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_DEG][block_col],
                        graph->blocks[block_row*bcount+block_col]->vcount
                    );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_update_dest(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                    problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                    options->damping,
                    graph->blocks_diag[block]->vcount
                );
        }

        for (graph_size_t block = 0; block < bcount; block++)
            cuda_pagerank_calc_diff(
                problem->streams[block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                graph->blocks_diag[block]->vcount
            );

        for (graph_size_t block = 0; block < bcount; block++)
            cuda_pagerank_sum_arr_offset(
                problem->streams[block],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_DIF],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                block,
                graph->blocks_diag[block]->vcount
            );

        cuda_pr_problem_synchronize(problem);

        PAGERANK_TIME_START(TRANSFER)
        pr_float tmp_diff[bcount];

        CUDA_ASSERT(cudaMemcpy(
            tmp_diff,
            problem->data_global_f[E_PR_PROBLEM_GLOBAL_DIF],
            sizeof(tmp_diff[0]) * bcount,
            cudaMemcpyDeviceToHost
        ));
        PAGERANK_TIME_STOP(TRANSFER)

        diff = cpu_pagerank_sum_arr(tmp_diff, bcount);

        iterations += options->local_iterations;
    }

    if (options->result != NULL)
    {
        PAGERANK_TIME_START(TRANSFER)
        for (graph_size_t block = 0; block < bcount; block++)
            cuda_pagerank_read_col(
                problem->streams[block],
                options->result,
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->blocks_diag[block]->vcount
            );

        cuda_pr_problem_synchronize(problem);
        PAGERANK_TIME_STOP(TRANSFER)
    }

    cuda_pr_problem_free(problem);
    return iterations;
}

uint32_t pagerank_bcsc_cuda_stepped(const pr_bcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    PAGERANK_TIME_START(TRANSFER)
    cuda_pr_problem_t *problem = cuda_pr_problem_new_bcsc(graph);
    PAGERANK_TIME_STOP(TRANSFER)

    const pr_float init = 1.0 / vcount;

    for (graph_size_t block = 0; block < bcount; block++)
        cuda_pagerank_fill_arr(
            problem->streams[block],
            problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
            init,
            graph->blocks_diag[block]->vcount
        );

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            cuda_pr_problem_swap_src_dst(problem);

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_baserank(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][block],
                    problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_DEG][block],
                    graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_sum_arr_offset(
                    problem->streams[block],
                    problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                    block,
                    graph->blocks_diag[block]->vcount
                );

            cuda_pr_problem_synchronize(problem);

            cuda_pagerank_sum_arr_offset(
                problem->streams[0],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                0,
                bcount
            );

            cuda_pagerank_baserank_redux(
                problem->streams[0],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                options->damping,
                vcount
            );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_update_tmp(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_VTX][block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][block],
                    problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_DEG][block],
                    graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_fill_arr(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                    0.0,
                    graph->blocks_diag[block]->vcount
                );

            cuda_pr_problem_synchronize(problem);

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                    cuda_pagerank_update_rank_tmp_pull(
                        problem->streams[block_row],
                        problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block_row],
                        problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_VTX][block_col],
                        problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_RID][block_row*bcount+block_col],
                        problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_CID][block_row*bcount+block_col],
                        graph->blocks[block_row*bcount+block_col]->vcount
                    );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_update_dest(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                    problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                    options->damping,
                    graph->blocks_diag[block]->vcount
                );

            cuda_pr_problem_synchronize(problem);
        }

        for (graph_size_t block = 0; block < bcount; block++)
            cuda_pagerank_calc_diff(
                problem->streams[block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                graph->blocks_diag[block]->vcount
            );

        for (graph_size_t block = 0; block < bcount; block++)
            cuda_pagerank_sum_arr_offset(
                problem->streams[block],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_DIF],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                block,
                graph->blocks_diag[block]->vcount
            );

        cuda_pr_problem_synchronize(problem);

        PAGERANK_TIME_START(TRANSFER)
        pr_float tmp_diff[bcount];

        CUDA_ASSERT(cudaMemcpy(
            &tmp_diff[0],
            problem->data_global_f[E_PR_PROBLEM_GLOBAL_DIF],
            sizeof(tmp_diff[0]) * bcount,
            cudaMemcpyDeviceToHost
        ));
        PAGERANK_TIME_STOP(TRANSFER)

        diff = cpu_pagerank_sum_arr(tmp_diff, bcount);

        iterations += options->local_iterations;
    }

    if (options->result != NULL)
    {
        PAGERANK_TIME_START(TRANSFER)
        for (graph_size_t block = 0; block < bcount; block++)
            cuda_pagerank_read_col(
                problem->streams[block],
                options->result,
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->blocks_diag[block]->vcount
            );

        cuda_pr_problem_synchronize(problem);
        PAGERANK_TIME_STOP(TRANSFER)
    }

    cuda_pr_problem_free(problem);
    return iterations;
}

uint32_t pagerank_bcsc_cuda_stepped_warp(const pr_bcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    PAGERANK_TIME_START(TRANSFER)
    cuda_pr_problem_t *problem = cuda_pr_problem_new_bcsc(graph);
    PAGERANK_TIME_STOP(TRANSFER)

    const pr_float init = 1.0 / vcount;

    for (graph_size_t block = 0; block < bcount; block++)
        cuda_pagerank_fill_arr(
            problem->streams[block],
            problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
            init,
            graph->blocks_diag[block]->vcount
        );

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            cuda_pr_problem_swap_src_dst(problem);

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_baserank(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][block],
                    problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_DEG][block],
                    graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_parallel_sum_arr_offset(
                    problem->streams[block],
                    problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_SCR][block],
                    block,
                    graph->blocks_diag[block]->vcount,
                    PAGERANK_SCRATCH_SIZE
                );

            cuda_pr_problem_synchronize(problem);

            cuda_pagerank_sum_arr_offset(
                problem->streams[0],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                0,
                bcount
            );

            cuda_pagerank_baserank_redux(
                problem->streams[0],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                options->damping,
                vcount
            );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_update_tmp(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_VTX][block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][block],
                    problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_DEG][block],
                    graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_fill_arr(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                    0.0,
                    graph->blocks_diag[block]->vcount
                );

            cuda_pr_problem_synchronize(problem);

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                    cuda_pagerank_update_rank_tmp_pull_warp(
                        problem->streams[block_row],
                        graph->blocks[block_row*bcount+block_col]->ecount / graph->blocks[block_row*bcount+block_col]->vcount,
                        problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block_row],
                        problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_VTX][block_col],
                        problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_RID][block_row*bcount+block_col],
                        problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_CID][block_row*bcount+block_col],
                        graph->blocks[block_row*bcount+block_col]->vcount
                    );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_update_dest(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                    problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                    options->damping,
                    graph->blocks_diag[block]->vcount
                );

            cuda_pr_problem_synchronize(problem);
        }

        for (graph_size_t block = 0; block < bcount; block++)
            cuda_pagerank_calc_diff(
                problem->streams[block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                graph->blocks_diag[block]->vcount
            );

        for (graph_size_t block = 0; block < bcount; block++)
            cuda_pagerank_parallel_sum_arr_offset(
                problem->streams[block],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_DIF],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_SCR][block],
                block,
                graph->blocks_diag[block]->vcount,
                PAGERANK_SCRATCH_SIZE
            );

        cuda_pr_problem_synchronize(problem);

        PAGERANK_TIME_START(TRANSFER)
        pr_float tmp_diff[bcount];

        CUDA_ASSERT(cudaMemcpy(
            tmp_diff,
            problem->data_global_f[E_PR_PROBLEM_GLOBAL_DIF],
            sizeof(tmp_diff[0]) * bcount,
            cudaMemcpyDeviceToHost
        ));
        PAGERANK_TIME_STOP(TRANSFER)

        diff = cpu_pagerank_sum_arr(tmp_diff, bcount);

        iterations += options->local_iterations;
    }

    if (options->result != NULL)
    {
        PAGERANK_TIME_START(TRANSFER)
        for (graph_size_t block = 0; block < bcount; block++)
            cuda_pagerank_read_col(
                problem->streams[block],
                options->result,
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->blocks_diag[block]->vcount
            );

        cuda_pr_problem_synchronize(problem);
        PAGERANK_TIME_STOP(TRANSFER)
    }

    cuda_pr_problem_free(problem);
    return iterations;
}

uint32_t pagerank_bcsc_cuda_stepped_dyn(const pr_bcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    PAGERANK_TIME_START(TRANSFER)
    cuda_pr_problem_t *problem = cuda_pr_problem_new_bcsc(graph);
    PAGERANK_TIME_STOP(TRANSFER)

    const pr_float init = 1.0 / vcount;

    for (graph_size_t block = 0; block < bcount; block++)
        cuda_pagerank_fill_arr(
            problem->streams[block],
            problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
            init,
            graph->blocks_diag[block]->vcount
        );

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            cuda_pr_problem_swap_src_dst(problem);

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_baserank(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][block],
                    problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_DEG][block],
                    graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_parallel_sum_arr_offset(
                    problem->streams[block],
                    problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_SCR][block],
                    block,
                    graph->blocks_diag[block]->vcount,
                    PAGERANK_SCRATCH_SIZE
                );

            cuda_pr_problem_synchronize(problem);

            cuda_pagerank_sum_arr_offset(
                problem->streams[0],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                0,
                bcount
            );

            cuda_pagerank_baserank_redux(
                problem->streams[0],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                options->damping,
                vcount
            );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_update_tmp(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_VTX][block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][block],
                    problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_DEG][block],
                    graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_fill_arr(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                    0.0,
                    graph->blocks_diag[block]->vcount
                );

            cuda_pr_problem_synchronize(problem);

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                    cuda_pagerank_update_rank_tmp_pull_dyn(
                        problem->streams[block_row],
                        graph->blocks[block_row*bcount+block_col]->ecount / graph->blocks[block_row*bcount+block_col]->vcount,
                        problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_TMP_SCR][block_row],
                        problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block_row],
                        problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_VTX][block_col],
                        problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_RID][block_row*bcount+block_col],
                        problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_CID][block_row*bcount+block_col],
                        graph->blocks[block_row*bcount+block_col]->vcount
                    );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_update_dest(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                    problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK],
                    options->damping,
                    graph->blocks_diag[block]->vcount
                );

            cuda_pr_problem_synchronize(problem);
        }

        for (graph_size_t block = 0; block < bcount; block++)
            cuda_pagerank_calc_diff(
                problem->streams[block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                graph->blocks_diag[block]->vcount
            );

        for (graph_size_t block = 0; block < bcount; block++)
            cuda_pagerank_parallel_sum_arr_offset(
                problem->streams[block],
                problem->data_global_f[E_PR_PROBLEM_GLOBAL_DIF],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_SCR][block],
                block,
                graph->blocks_diag[block]->vcount,
                PAGERANK_SCRATCH_SIZE
            );

        cuda_pr_problem_synchronize(problem);

        PAGERANK_TIME_START(TRANSFER)
        pr_float tmp_diff[bcount];

        CUDA_ASSERT(cudaMemcpy(
            tmp_diff,
            problem->data_global_f[E_PR_PROBLEM_GLOBAL_DIF],
            sizeof(tmp_diff[0]) * bcount,
            cudaMemcpyDeviceToHost
        ));
        PAGERANK_TIME_STOP(TRANSFER)

        diff = cpu_pagerank_sum_arr(tmp_diff, bcount);

        iterations += options->local_iterations;
    }

    if (options->result != NULL)
    {
        PAGERANK_TIME_START(TRANSFER)
        for (graph_size_t block = 0; block < bcount; block++)
            cuda_pagerank_read_col(
                problem->streams[block],
                options->result,
                problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->blocks_diag[block]->vcount
            );

        cuda_pr_problem_synchronize(problem);
        PAGERANK_TIME_STOP(TRANSFER)
    }

    cuda_pr_problem_free(problem);
    return iterations;
}

uint32_t pagerank_bcsc_cuda_stepped_mix(const pr_bcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    PAGERANK_TIME_START(TRANSFER)
    cuda_pr_problem_t *problem = cuda_pr_problem_new_bcsc(graph);
    PAGERANK_TIME_STOP(TRANSFER)

    PAGERANK_TIME_START(INIT)
    pr_float *restrict *restrict src = memory_talloc(pr_float*, bcount);
    pr_float *restrict *restrict dst = memory_talloc(pr_float*, bcount);

    const pr_float init = 1.0 / vcount;

    OMP_PRAGMA(omp parallel for)
    for (graph_size_t block = 0; block < bcount; block++)
    {
        const graph_size_t block_len = graph->blocks_diag[block]->vcount;
        src[block] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));
        dst[block] = memory_talloc(pr_float, ROUND_TO_MULT(block_len, BCSR_GRAPH_VERTEX_PACK));

        cuda_pagerank_fill_arr(
            problem->streams[block],
            problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
            init,
            block_len
        );
        cpu_pagerank_fill_arr(dst[block], init, block_len);
    }
    PAGERANK_TIME_STOP(INIT)

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            SWAP_VALUES(src, dst);
            cuda_pr_problem_swap_src_dst(problem);

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_update_tmp(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_VTX][block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][block],
                    problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_DEG][block],
                    graph->blocks_diag[block]->vcount
                );

            for (graph_size_t block = 0; block < bcount; block++)
                cuda_pagerank_fill_arr(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                    0.0,
                    graph->blocks_diag[block]->vcount
                );

            PAGERANK_TIME_START(BASERANK)
            pr_float base_rank = 0.0;

            OMP_PRAGMA(omp parallel for reduction(+:base_rank))
            for (graph_size_t block = 0; block < bcount; block++)
                base_rank += cpu_pagerank_baserank(
                    src[block],
                    graph->blocks_diag[block]->deg_i,
                    graph->blocks_diag[block]->vcount
                );

            base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);
            PAGERANK_TIME_STOP(BASERANK)

            cuda_pr_problem_synchronize(problem);

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                    cuda_pagerank_update_rank_tmp_pull_warp(
                        problem->streams[block_row],
                        graph->blocks[block_row*bcount+block_col]->ecount / graph->blocks[block_row*bcount+block_col]->vcount,
                        problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block_row],
                        problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_VTX][block_col],
                        problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_RID][block_row*bcount+block_col],
                        problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_CID][block_row*bcount+block_col],
                        graph->blocks[block_row*bcount+block_col]->vcount
                    );

            for (graph_size_t block = 0; block < bcount; block++)
            {
                cuda_pagerank_update_dest_raw(
                    problem->streams[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                    base_rank,
                    options->damping,
                    graph->blocks_diag[block]->vcount
                );
                CUDA_ASSERT(cudaMemcpyAsync(
                    dst[block],
                    problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][block],
                    sizeof(*dst[block]) * graph->blocks_diag[block]->vcount,
                    cudaMemcpyDeviceToHost,
                    problem->streams[block]
                ));
            }

            cuda_pr_problem_synchronize(problem);
        }

        PAGERANK_TIME_START(DIFF)
        diff = 0.0;

        OMP_PRAGMA(omp parallel for reduction(+:diff))
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
        OMP_PRAGMA(omp parallel for)
        for (graph_size_t block = 0; block < bcount; block++)
            cpu_pagerank_read_col(
                options->result,
                dst[block],
                BCSR_GRAPH_VERTEX_PACK * block,
                BCSR_GRAPH_VERTEX_PACK * bcount,
                BCSR_GRAPH_VERTEX_PACK,
                graph->blocks_diag[block]->vcount
            );

    for (graph_size_t block = 0; block < bcount; block++)
    {
        memory_free((void*)src[block]);
        memory_free((void*)dst[block]);
    }

    memory_free((void*)src);
    memory_free((void*)dst);

    cuda_pr_problem_free(problem);
    return iterations;
}
