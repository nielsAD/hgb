// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/ocl_codelets.h"
#include "alg/pagerank/cpu_codelets.h"
#include "alg/pagerank/ocl_problem.h"
#include "util/math.h"
#include "util/memory.h"

static inline void ocl_pr_problem_swap_src_dst(ocl_pr_problem_t *problem)
{
    const graph_size_t bcount = problem->graph->bcount;

    for (graph_size_t b = 0; b < bcount*bcount; b++)
    {
        SWAP_VALUES(problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][b], problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][b]);
        SWAP_VALUES(problem->event_blocks[E_PR_PROBLEM_BLOCKS_SRC][b], problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][b]);
    }
}

uint32_t pagerank_bcsc_ocl_mapped(const pr_bcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    PAGERANK_TIME_START(TRANSFER)
    ocl_pr_problem_t *problem = ocl_pr_problem_new_bcsc(options->devid, graph);
    PAGERANK_TIME_STOP(TRANSFER)

    const pr_float init = 1.0 / vcount;

    for (graph_size_t block = 0; block < bcount; block++)
        ocl_pr_problem_set_event_b(problem,
            E_PR_PROBLEM_BLOCKS_DST,
            block,
            ocl_pagerank_fill_arr(
                options->devid,
                problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                init,
                graph->blocks_diag[block]->vcount,
                0, NULL
            )
        );

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            ocl_pr_problem_swap_src_dst(problem);

            for (graph_size_t block = 0; block < bcount; block++)
                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_TMP_RNK,
                    block,
                    ocl_pagerank_baserank(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][block],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][block],
                        graph->blocks_diag[block]->vcount,
                        1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_SRC][block]
                    )
                );

            for (graph_size_t block = 0; block < bcount; block++)
                ocl_pr_problem_set_event_g(problem,
                    E_PR_PROBLEM_GLOBAL_RNK,
                    block,
                    ocl_pagerank_sum_arr_offset(
                        options->devid,
                        problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                        block,
                        graph->blocks_diag[block]->vcount,
                        1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_TMP_RNK][block]
                    )
                );

            ocl_pr_problem_set_event_g(problem,
                E_PR_PROBLEM_GLOBAL_RNK,
                0,
                ocl_pagerank_sum_arr_offset(
                    options->devid,
                    problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                    problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                    0,
                    bcount,
                    bcount, &problem->event_global[E_PR_PROBLEM_GLOBAL_RNK][0]
                )
            );

            ocl_pr_problem_set_event_g(problem,
                E_PR_PROBLEM_GLOBAL_RNK,
                0,
                ocl_pagerank_baserank_redux(
                    options->devid,
                    problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                    options->damping,
                    vcount,
                    1, &problem->event_global[E_PR_PROBLEM_GLOBAL_RNK][0]
                )
            );

            for (graph_size_t block = 0; block < bcount; block++)
                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_DST,
                    block,
                    ocl_pagerank_fill_arr(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                        0.0,
                        graph->blocks_diag[block]->vcount,
                        (local_iterations == 0) ? 0 : 1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block]
                    )
                );

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                {
                    cl_event events[2];
                    events[0] = problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block_row];
                    events[1] = problem->event_blocks[E_PR_PROBLEM_BLOCKS_SRC][block_col];

                    ocl_pr_problem_set_event_b(problem,
                        E_PR_PROBLEM_BLOCKS_DST,
                        block_row,
                        ocl_pagerank_update_rank_pull(
                            options->devid,
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block_row],
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][block_col],
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_RID][block_row*bcount+block_col],
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_CID][block_row*bcount+block_col],
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][block_col],
                            graph->blocks[block_row*bcount+block_col]->vcount,
                            2, events
                        )
                    );
                }

            for (graph_size_t block = 0; block < bcount; block++)
            {
                cl_event events[2];
                events[0] = problem->event_global[E_PR_PROBLEM_GLOBAL_RNK][0];
                events[1] = problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block];

                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_DST,
                    block,
                    ocl_pagerank_update_dest(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                        problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                        options->damping,
                        graph->blocks_diag[block]->vcount,
                        2, events
                    )
                );
            }
        }

        for (graph_size_t block = 0; block < bcount; block++)
            ocl_pr_problem_set_event_b(problem,
                E_PR_PROBLEM_BLOCKS_TMP_DIF,
                block,
                ocl_pagerank_calc_diff(
                    options->devid,
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][block],
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                    graph->blocks_diag[block]->vcount,
                    1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block]
                )
            );

        for (graph_size_t block = 0; block < bcount; block++)
            ocl_pr_problem_set_event_g(problem,
                E_PR_PROBLEM_GLOBAL_DIF,
                block,
                ocl_pagerank_sum_arr_offset(
                    options->devid,
                    problem->data_global[E_PR_PROBLEM_GLOBAL_DIF],
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                    block,
                    graph->blocks_diag[block]->vcount,
                    1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_TMP_DIF][block]
                )
            );

        OPENCL_ASSERT(clWaitForEvents(bcount, &problem->event_global[E_PR_PROBLEM_GLOBAL_DIF][0]));

        PAGERANK_TIME_START(TRANSFER)
        pr_float tmp_diff[bcount];

        OPENCL_ASSERT(clEnqueueReadBuffer(
            problem->queue,
            problem->data_global[E_PR_PROBLEM_GLOBAL_DIF],
            CL_TRUE,
            0, sizeof(tmp_diff[0]) * bcount,
            tmp_diff,
            bcount, &problem->event_global[E_PR_PROBLEM_GLOBAL_DIF][0],
            NULL
        ));
        PAGERANK_TIME_STOP(TRANSFER)

        diff = cpu_pagerank_sum_arr(tmp_diff, bcount);

        iterations += options->local_iterations;
    }

    if (options->result != NULL)
    {
        PAGERANK_TIME_START(TRANSFER)
        for (graph_size_t block = 0; block < bcount; block++)
            ocl_pr_problem_set_event_b(problem,
                E_PR_PROBLEM_BLOCKS_DST,
                block,
                ocl_pagerank_read_col(
                    options->devid,
                    options->result,
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                    BCSR_GRAPH_VERTEX_PACK * block,
                    BCSR_GRAPH_VERTEX_PACK * bcount,
                    BCSR_GRAPH_VERTEX_PACK,
                    graph->blocks_diag[block]->vcount,
                    0, NULL
                )
            );

        OPENCL_ASSERT(clWaitForEvents(bcount, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][0]));
        PAGERANK_TIME_STOP(TRANSFER)
    }

    ocl_pr_problem_free(problem);
    return iterations;
}

uint32_t pagerank_bcsc_ocl_stepped(const pr_bcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    PAGERANK_TIME_START(TRANSFER)
    ocl_pr_problem_t *problem = ocl_pr_problem_new_bcsc(options->devid, graph);
    PAGERANK_TIME_STOP(TRANSFER)

    const pr_float init = 1.0 / vcount;

    for (graph_size_t block = 0; block < bcount; block++)
        ocl_pr_problem_set_event_b(problem,
            E_PR_PROBLEM_BLOCKS_DST,
            block,
            ocl_pagerank_fill_arr(
                options->devid,
                problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                init,
                graph->blocks_diag[block]->vcount,
                0, NULL
            )
        );

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            ocl_pr_problem_swap_src_dst(problem);

            for (graph_size_t block = 0; block < bcount; block++)
                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_TMP_RNK,
                    block,
                    ocl_pagerank_baserank(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][block],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][block],
                        graph->blocks_diag[block]->vcount,
                        1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_SRC][block]
                    )
                );

            for (graph_size_t block = 0; block < bcount; block++)
                ocl_pr_problem_set_event_g(problem,
                    E_PR_PROBLEM_GLOBAL_RNK,
                    block,
                    ocl_pagerank_sum_arr_offset(
                        options->devid,
                        problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                        block,
                        graph->blocks_diag[block]->vcount,
                        1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_TMP_RNK][block]
                    )
                );


            ocl_pr_problem_set_event_g(problem,
                E_PR_PROBLEM_GLOBAL_RNK,
                0,
                ocl_pagerank_sum_arr_offset(
                    options->devid,
                    problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                    problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                    0,
                    bcount,
                    bcount, &problem->event_global[E_PR_PROBLEM_GLOBAL_RNK][0]
                )
            );

            ocl_pr_problem_set_event_g(problem,
                E_PR_PROBLEM_GLOBAL_RNK,
                0,
                ocl_pagerank_baserank_redux(
                    options->devid,
                    problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                    options->damping,
                    vcount,
                    1, &problem->event_global[E_PR_PROBLEM_GLOBAL_RNK][0]
                )
            );

            for (graph_size_t block = 0; block < bcount; block++)
                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_TMP_VTX,
                    block,
                    ocl_pagerank_update_tmp(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][block],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][block],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][block],
                        graph->blocks_diag[block]->vcount,
                        1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_SRC][block]
                    )
                );

            for (graph_size_t block = 0; block < bcount; block++)
                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_DST,
                    block,
                    ocl_pagerank_fill_arr(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                        0.0,
                        graph->blocks_diag[block]->vcount,
                        (local_iterations == 0) ? 0 : 1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block]
                    )
                );

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                {
                    cl_event events[2];
                    events[0] = problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block_row];
                    events[1] = problem->event_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][block_col];

                    ocl_pr_problem_set_event_b(problem,
                        E_PR_PROBLEM_BLOCKS_DST,
                        block_row,
                        ocl_pagerank_update_rank_tmp_pull(
                            options->devid,
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block_row],
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][block_col],
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_RID][block_row*bcount+block_col],
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_CID][block_row*bcount+block_col],
                            graph->blocks[block_row*bcount+block_col]->vcount,
                            2, events
                        )
                    );
                }

            for (graph_size_t block = 0; block < bcount; block++)
            {
                cl_event events[2];
                events[0] = problem->event_global[E_PR_PROBLEM_GLOBAL_RNK][0];
                events[1] = problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block];

                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_DST,
                    block,
                    ocl_pagerank_update_dest(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                        problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                        options->damping,
                        graph->blocks_diag[block]->vcount,
                        2, events
                    )
                );
            }
        }

        for (graph_size_t block = 0; block < bcount; block++)
            ocl_pr_problem_set_event_b(problem,
                E_PR_PROBLEM_BLOCKS_TMP_DIF,
                block,
                ocl_pagerank_calc_diff(
                    options->devid,
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][block],
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                    graph->blocks_diag[block]->vcount,
                    1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block]
                )
            );

        for (graph_size_t block = 0; block < bcount; block++)
            ocl_pr_problem_set_event_g(problem,
                E_PR_PROBLEM_GLOBAL_DIF,
                block,
                ocl_pagerank_sum_arr_offset(
                    options->devid,
                    problem->data_global[E_PR_PROBLEM_GLOBAL_DIF],
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                    block,
                    graph->blocks_diag[block]->vcount,
                    1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_TMP_DIF][block]
                )
            );

        OPENCL_ASSERT(clWaitForEvents(bcount, &problem->event_global[E_PR_PROBLEM_GLOBAL_DIF][0]));

        PAGERANK_TIME_START(TRANSFER)
        pr_float tmp_diff[bcount];

        OPENCL_ASSERT(clEnqueueReadBuffer(
            problem->queue,
            problem->data_global[E_PR_PROBLEM_GLOBAL_DIF],
            CL_TRUE,
            0, sizeof(tmp_diff[0]) * bcount,
            tmp_diff,
            bcount, &problem->event_global[E_PR_PROBLEM_GLOBAL_DIF][0],
            NULL
        ));
        PAGERANK_TIME_STOP(TRANSFER)

        diff = cpu_pagerank_sum_arr(tmp_diff, bcount);

        iterations += options->local_iterations;
    }

    if (options->result != NULL)
    {
        PAGERANK_TIME_START(TRANSFER)
        for (graph_size_t block = 0; block < bcount; block++)
            ocl_pr_problem_set_event_b(problem,
                E_PR_PROBLEM_BLOCKS_DST,
                block,
                ocl_pagerank_read_col(
                    options->devid,
                    options->result,
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                    BCSR_GRAPH_VERTEX_PACK * block,
                    BCSR_GRAPH_VERTEX_PACK * bcount,
                    BCSR_GRAPH_VERTEX_PACK,
                    graph->blocks_diag[block]->vcount,
                    0, NULL
                )
            );

        OPENCL_ASSERT(clWaitForEvents(bcount, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][0]));
        PAGERANK_TIME_STOP(TRANSFER)
    }

    ocl_pr_problem_free(problem);
    return iterations;
}

uint32_t pagerank_bcsc_ocl_stepped_warp(const pr_bcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    PAGERANK_TIME_START(TRANSFER)
    ocl_pr_problem_t *problem = ocl_pr_problem_new_bcsc(options->devid, graph);
    PAGERANK_TIME_STOP(TRANSFER)

    const pr_float init = 1.0 / vcount;

    for (graph_size_t block = 0; block < bcount; block++)
        ocl_pr_problem_set_event_b(problem,
            E_PR_PROBLEM_BLOCKS_DST,
            block,
            ocl_pagerank_fill_arr(
                options->devid,
                problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                init,
                graph->blocks_diag[block]->vcount,
                0, NULL
            )
        );

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            ocl_pr_problem_swap_src_dst(problem);

            for (graph_size_t block = 0; block < bcount; block++)
                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_TMP_RNK,
                    block,
                    ocl_pagerank_baserank(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][block],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][block],
                        graph->blocks_diag[block]->vcount,
                        1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_SRC][block]
                    )
                );

            for (graph_size_t block = 0; block < bcount; block++)
                ocl_pr_problem_set_event_g(problem,
                    E_PR_PROBLEM_GLOBAL_RNK,
                    block,
                    ocl_pagerank_parallel_sum_arr_offset(
                        options->devid,
                        problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_RNK][block],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_SCR][block],
                        block,
                        graph->blocks_diag[block]->vcount,
                        PAGERANK_SCRATCH_SIZE,
                        1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_TMP_RNK][block]
                    )
                );


            ocl_pr_problem_set_event_g(problem,
                E_PR_PROBLEM_GLOBAL_RNK,
                0,
                ocl_pagerank_sum_arr_offset(
                    options->devid,
                    problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                    problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                    0,
                    bcount,
                    bcount, &problem->event_global[E_PR_PROBLEM_GLOBAL_RNK][0]
                )
            );

            ocl_pr_problem_set_event_g(problem,
                E_PR_PROBLEM_GLOBAL_RNK,
                0,
                ocl_pagerank_baserank_redux(
                    options->devid,
                    problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                    options->damping,
                    vcount,
                    1, &problem->event_global[E_PR_PROBLEM_GLOBAL_RNK][0]
                )
            );

            for (graph_size_t block = 0; block < bcount; block++)
                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_TMP_VTX,
                    block,
                    ocl_pagerank_update_tmp(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][block],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][block],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][block],
                        graph->blocks_diag[block]->vcount,
                        1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_SRC][block]
                    )
                );

            for (graph_size_t block = 0; block < bcount; block++)
                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_DST,
                    block,
                    ocl_pagerank_fill_arr(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                        0.0,
                        graph->blocks_diag[block]->vcount,
                        (local_iterations == 0) ? 0 : 1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block]
                    )
                );

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                {
                    cl_event events[2];
                    events[0] = problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block_row];
                    events[1] = problem->event_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][block_col];

                    ocl_pr_problem_set_event_b(problem,
                        E_PR_PROBLEM_BLOCKS_DST,
                        block_row,
                        ocl_pagerank_update_rank_tmp_pull_warp(
                            options->devid,
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block_row],
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][block_col],
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_RID][block_row*bcount+block_col],
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_CID][block_row*bcount+block_col],
                            graph->blocks[block_row*bcount+block_col]->vcount,
                            2, events
                        )
                    );
                }

            for (graph_size_t block = 0; block < bcount; block++)
            {
                cl_event events[2];
                events[0] = problem->event_global[E_PR_PROBLEM_GLOBAL_RNK][0];
                events[1] = problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block];

                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_DST,
                    block,
                    ocl_pagerank_update_dest(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                        problem->data_global[E_PR_PROBLEM_GLOBAL_RNK],
                        options->damping,
                        graph->blocks_diag[block]->vcount,
                        2, events
                    )
                );
            }
        }

        for (graph_size_t block = 0; block < bcount; block++)
            ocl_pr_problem_set_event_b(problem,
                E_PR_PROBLEM_BLOCKS_TMP_DIF,
                block,
                ocl_pagerank_calc_diff(
                    options->devid,
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][block],
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                    graph->blocks_diag[block]->vcount,
                    1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block]
                )
            );

        for (graph_size_t block = 0; block < bcount; block++)
            ocl_pr_problem_set_event_g(problem,
                E_PR_PROBLEM_GLOBAL_DIF,
                block,
                ocl_pagerank_parallel_sum_arr_offset(
                    options->devid,
                    problem->data_global[E_PR_PROBLEM_GLOBAL_DIF],
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DIF][block],
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_SCR][block],
                    block,
                    graph->blocks_diag[block]->vcount,
                    PAGERANK_SCRATCH_SIZE,
                    1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_TMP_DIF][block]
                )
            );

        OPENCL_ASSERT(clWaitForEvents(bcount, &problem->event_global[E_PR_PROBLEM_GLOBAL_DIF][0]));

        PAGERANK_TIME_START(TRANSFER)
        pr_float tmp_diff[bcount];

        OPENCL_ASSERT(clEnqueueReadBuffer(
            problem->queue,
            problem->data_global[E_PR_PROBLEM_GLOBAL_DIF],
            CL_TRUE,
            0, sizeof(tmp_diff[0]) * bcount,
            tmp_diff,
            bcount, &problem->event_global[E_PR_PROBLEM_GLOBAL_DIF][0],
            NULL
        ));
        PAGERANK_TIME_STOP(TRANSFER)

        diff = cpu_pagerank_sum_arr(tmp_diff, bcount);

        iterations += options->local_iterations;
    }

    if (options->result != NULL)
    {
        PAGERANK_TIME_START(TRANSFER)
        for (graph_size_t block = 0; block < bcount; block++)
            ocl_pr_problem_set_event_b(problem,
                E_PR_PROBLEM_BLOCKS_DST,
                block,
                ocl_pagerank_read_col(
                    options->devid,
                    options->result,
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                    BCSR_GRAPH_VERTEX_PACK * block,
                    BCSR_GRAPH_VERTEX_PACK * bcount,
                    BCSR_GRAPH_VERTEX_PACK,
                    graph->blocks_diag[block]->vcount,
                    0, NULL
                )
            );

        OPENCL_ASSERT(clWaitForEvents(bcount, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][0]));
        PAGERANK_TIME_STOP(TRANSFER)
    }

    ocl_pr_problem_free(problem);
    return iterations;
}

uint32_t pagerank_bcsc_ocl_stepped_mix(const pr_bcsc_graph_t *graph, pagerank_options_t *options)
{
    assert(graph != NULL);

    const graph_size_t bcount = graph->bcount;
    const graph_size_t vcount = graph->vcount;

    PAGERANK_TIME_START(TRANSFER)
    ocl_pr_problem_t *problem = ocl_pr_problem_new_bcsc(options->devid, graph);
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

        cpu_pagerank_fill_arr(dst[block], init, block_len);
    }
    PAGERANK_TIME_STOP(INIT)

    for (graph_size_t block = 0; block < bcount; block++)
    {
        ocl_pr_problem_set_event_b(problem,
            E_PR_PROBLEM_BLOCKS_DST,
            block,
            ocl_pagerank_fill_arr(
                options->devid,
                problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                init,
                graph->blocks_diag[block]->vcount,
                0, NULL
            )
        );
    }

    uint32_t iterations = 0;
    pr_float diff       = 1.0;

    while ((iterations < options->min_iterations) || (iterations < options->max_iterations && diff > options->epsilon))
    {
        for (uint32_t local_iterations = 0; local_iterations < options->local_iterations; local_iterations++)
        {
            SWAP_VALUES(src, dst);
            ocl_pr_problem_swap_src_dst(problem);

            for (graph_size_t block = 0; block < bcount; block++)
                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_TMP_VTX,
                    block,
                    ocl_pagerank_update_tmp(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][block],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][block],
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][block],
                        graph->blocks_diag[block]->vcount,
                        1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_SRC][block]
                    )
                );

            for (graph_size_t block = 0; block < bcount; block++)
                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_DST,
                    block,
                    ocl_pagerank_fill_arr(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                        0.0,
                        graph->blocks_diag[block]->vcount,
                        (local_iterations == 0) ? 0 : 1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block]
                    )
                );

            for (graph_size_t block_col = 0; block_col < bcount; block_col++)
                for (graph_size_t block_row = 0; block_row < bcount; block_row++)
                {
                    cl_event events[2];
                    events[0] = problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block_row];
                    events[1] = problem->event_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][block_col];

                    ocl_pr_problem_set_event_b(problem,
                        E_PR_PROBLEM_BLOCKS_DST,
                        block_row,
                        ocl_pagerank_update_rank_tmp_pull_warp(
                            options->devid,
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block_row],
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][block_col],
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_RID][block_row*bcount+block_col],
                            problem->data_blocks[E_PR_PROBLEM_BLOCKS_CID][block_row*bcount+block_col],
                            graph->blocks[block_row*bcount+block_col]->vcount,
                            2, events
                        )
                    );
                }

            pr_float base_rank = 0.0;

            OMP_PRAGMA(omp parallel for reduction(+:base_rank))
            for (graph_size_t block = 0; block < bcount; block++)
                base_rank += cpu_pagerank_baserank(
                    src[block],
                    graph->blocks_diag[block]->deg_i,
                    graph->blocks_diag[block]->vcount
                );

            base_rank = cpu_pagerank_baserank_redux(base_rank, options->damping, vcount);

            for (graph_size_t block = 0; block < bcount; block++)
            {
                ocl_pr_problem_set_event_b(problem,
                    E_PR_PROBLEM_BLOCKS_DST,
                    block,
                    ocl_pagerank_update_dest_raw(
                        options->devid,
                        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                        base_rank,
                        options->damping,
                        graph->blocks_diag[block]->vcount,
                        1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block]
                    )
                );

                cl_event event;
                OPENCL_ASSERT(clEnqueueReadBuffer(
                    problem->queue,
                    problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                    CL_FALSE,
                    0, sizeof(*dst[block]) * graph->blocks_diag[block]->vcount,
                    dst[block],
                    1, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
                    &event
                ));
                ocl_pr_problem_set_event_b(problem, E_PR_PROBLEM_BLOCKS_DST, block, event);
            }

            OPENCL_ASSERT(clWaitForEvents(bcount, &problem->event_blocks[E_PR_PROBLEM_BLOCKS_DST][0]));
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

    ocl_pr_problem_free(problem);
    return iterations;
}
