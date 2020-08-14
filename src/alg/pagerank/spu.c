// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/spu_problem.h"
#include "alg/pagerank/spu_codelets.h"

static void starpu_task_invalidate_all(
    spu_pr_problem_t *problem,
    const pagerank_problem_data_block_enum_t arr)
{
    assert(problem != NULL);
    assert(problem->graph != NULL);

    for (graph_size_t idx = 0; idx < problem->graph->bcount; idx++)
        starpu_data_invalidate_submit(problem->data_blocks[arr][idx]);
}

static struct starpu_task *starpu_task_pagerank_read_col(
    pr_float *dst,
    const starpu_data_handle_t src,
    const graph_size_t dst_offset,
    const graph_size_t dst_cols,
    const graph_size_t src_cols,
    const graph_size_t rows)
{
    struct starpu_task *task = starpu_task_create();
    assert(task != NULL);

    task->cl = &cl_pagerank_read_col;
    task->cl_arg_free = true;

    starpu_codelet_pack_args(&task->cl_arg, &task->cl_arg_size,
        STARPU_VALUE, &dst,        sizeof(dst),
        STARPU_VALUE, &dst_offset, sizeof(dst_offset),
        STARPU_VALUE, &dst_cols,   sizeof(dst_cols),
        STARPU_VALUE, &src_cols,   sizeof(src_cols),
        STARPU_VALUE, &rows,       sizeof(rows),
        0
    );

    task->handles[0] = src;

    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
    return task;
}

static struct starpu_task *starpu_task_pagerank_fill_arr(
    const starpu_data_handle_t arr,
    const pr_float init)
{
    struct starpu_task *task = starpu_task_create();
    assert(task != NULL);

    task->cl = &cl_pagerank_fill_arr;
    task->cl_arg_free = true;

    starpu_codelet_pack_args(&task->cl_arg, &task->cl_arg_size,
        STARPU_VALUE, &init, sizeof(init),
        0
    );

    task->handles[0] = arr;

    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
    return task;
}

static void starpu_task_fill_all(
    spu_pr_problem_t *problem,
    const pagerank_problem_data_block_enum_t arr,
    const pr_float init)
{
    assert(problem != NULL);
    assert(problem->graph != NULL);

    for (graph_size_t idx = 0; idx < problem->graph->bcount; idx++)
        starpu_task_pagerank_fill_arr(problem->data_blocks[arr][idx], init);
}

static struct starpu_task *starpu_task_redux_sum(
    const starpu_data_handle_t res,
    const starpu_data_handle_t arr,
    const starpu_data_handle_t scr)
{
    struct starpu_task *task = starpu_task_create();
    assert(task != NULL);

    task->cl = &cl_pagerank_redux_sum;

    task->handles[0] = res;
    task->handles[1] = arr;
    task->handles[2] = scr;

    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
    return task;
}

static void starpu_task_redux_sum_all(
    spu_pr_problem_t *problem,
    const pagerank_problem_data_global_enum_t res,
    const pagerank_problem_data_block_enum_t arr)
{
    assert(problem != NULL);
    assert(problem->graph != NULL);

    for (graph_size_t idx = 0; idx < problem->graph->bcount; idx++)
        starpu_task_redux_sum(problem->data_global[res], problem->data_blocks[arr][idx], problem->data_global[E_PR_PROBLEM_GLOBAL_SCR]);
}

static struct starpu_task *starpu_task_pagerank(spu_pr_problem_t *p, struct starpu_codelet *cl)
{
    struct starpu_task *task = starpu_task_create();
    assert(task != NULL);

    task->cl = cl;

    task->cl_arg = p;
    task->cl_arg_size = sizeof(spu_pr_problem_t);
    task->cl_arg_free = false;

    return task;
}

static void starpu_task_pagerank_initialize(spu_pr_problem_t *problem)
{
    const pr_float init = 1.0 / problem->graph->vcount;
    starpu_task_fill_all(problem, E_PR_PROBLEM_BLOCKS_DST, init);
}

static void starpu_task_pagerank_finalize(spu_pr_problem_t *problem)
{
    const graph_size_t bcount = problem->graph->bcount;

    for (graph_size_t block = 0; block < bcount; block++)
        starpu_task_pagerank_read_col(
            problem->options->result,
            problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block],
            BCSR_GRAPH_VERTEX_PACK * block,
            BCSR_GRAPH_VERTEX_PACK * bcount,
            BCSR_GRAPH_VERTEX_PACK,
            problem->graph->blocks_diag[block]->vcount
        );

    starpu_task_wait_for_all();
}

static void starpu_task_pagerank_baserank(spu_pr_problem_t *problem)
{
    const graph_size_t bcount = problem->graph->bcount;

    for (graph_size_t block = 0; block < bcount; block++)
    {
        struct starpu_task *task = starpu_task_pagerank(problem, &cl_pagerank_baserank);

        task->handles[0] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_RNK][block];
        task->handles[1] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][block];
        task->handles[2] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][block];

        STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
    }

    starpu_task_redux_sum_all(problem, E_PR_PROBLEM_GLOBAL_RNK, E_PR_PROBLEM_BLOCKS_TMP_RNK);
    starpu_task_invalidate_all(problem, E_PR_PROBLEM_BLOCKS_TMP_RNK);
}

static void starpu_task_pagerank_baserank_redux(spu_pr_problem_t *problem)
{
    struct starpu_task *task = starpu_task_pagerank(problem, &cl_pagerank_baserank_redux);

    task->handles[0] = problem->data_global[E_PR_PROBLEM_GLOBAL_RNK];

    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
}

static void starpu_task_pagerank_update_rank_pull(spu_pr_problem_t *problem)
{
    const pr_bcsr_graph_t *graph  = problem->graph;
    const graph_size_t     bcount = graph->bcount;

    starpu_task_fill_all(problem, E_PR_PROBLEM_BLOCKS_DST, 0.0);

    for (graph_size_t row = 0; row < bcount; row++)
        for (graph_size_t col = 0; col < bcount; col++)
            if (graph->blocks[(row * bcount) + col]->ecount > 0)
            {
                struct starpu_task *task = starpu_task_pagerank(problem, &cl_pagerank_update_rank_pull);

                task->handles[0] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][row];
                task->handles[1] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][col];
                task->handles[2] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][col];
                task->handles[3] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_RID][(row * bcount) + col];
                task->handles[4] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_CID][(row * bcount) + col];

                STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
            }
}

static void starpu_task_pagerank_update_rank_push(spu_pr_problem_t *problem)
{
    const pr_bcsr_graph_t *graph  = problem->graph;
    const graph_size_t     bcount = graph->bcount;

    starpu_task_fill_all(problem, E_PR_PROBLEM_BLOCKS_DST, 0.0);

    for (graph_size_t row = 0; row < bcount; row++)
        for (graph_size_t col = 0; col < bcount; col++)
            if (graph->blocks[(row * bcount) + col]->ecount > 0)
            {
                struct starpu_task *task = starpu_task_pagerank(problem, &cl_pagerank_update_rank_push);

                task->handles[0] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][col];
                task->handles[1] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][row];
                task->handles[2] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][row];
                task->handles[3] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_RID][(row * bcount) + col];
                task->handles[4] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_CID][(row * bcount) + col];

                STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
            }
}

static void starpu_task_pagerank_update_tmp(spu_pr_problem_t *problem)
{
    const graph_size_t bcount = problem->graph->bcount;

    for (graph_size_t block = 0; block < bcount; block++)
    {
        struct starpu_task *task = starpu_task_pagerank(problem, &cl_pagerank_update_tmp);

        task->handles[0] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][block];
        task->handles[1] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][block];
        task->handles[2] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][block];

        STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
    }
}

static void starpu_task_pagerank_update_rank_tmp_pull(spu_pr_problem_t *problem)
{
    const pr_bcsr_graph_t *graph = problem->graph;
    const graph_size_t    bcount = graph->bcount;

    starpu_task_fill_all(problem, E_PR_PROBLEM_BLOCKS_DST, 0.0);

    for (graph_size_t row = 0; row < bcount; row++)
        for (graph_size_t col = 0; col < bcount; col++)
            if (graph->blocks[(row * bcount) + col]->ecount > 0)
            {
                struct starpu_task *task = starpu_task_pagerank(problem, &cl_pagerank_update_rank_tmp_pull);

                task->handles[0] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][row];
                task->handles[1] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][col];
                task->handles[2] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_RID][(row * bcount) + col];
                task->handles[3] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_CID][(row * bcount) + col];
                task->handles[4] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_SCR][row];

                STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
            }

    starpu_task_invalidate_all(problem, E_PR_PROBLEM_BLOCKS_TMP_VTX);
}

static void starpu_task_pagerank_update_rank_tmp_push(spu_pr_problem_t *problem)
{
    const pr_bcsr_graph_t *graph = problem->graph;
    const graph_size_t    bcount = graph->bcount;

    starpu_task_fill_all(problem, E_PR_PROBLEM_BLOCKS_DST, 0.0);

    for (graph_size_t row = 0; row < bcount; row++)
        for (graph_size_t col = 0; col < bcount; col++)
            if (graph->blocks[(row * bcount) + col]->ecount > 0)
            {
                struct starpu_task *task = starpu_task_pagerank(problem, &cl_pagerank_update_rank_tmp_push);

                task->handles[0] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][col];
                task->handles[1] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][row];
                task->handles[2] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_RID][(row * bcount) + col];
                task->handles[3] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_CID][(row * bcount) + col];
                task->handles[4] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_SCR][col];

                STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
            }

    starpu_task_invalidate_all(problem, E_PR_PROBLEM_BLOCKS_TMP_VTX);
}

static void starpu_task_pagerank_redux_rank_tmp_pull(spu_pr_problem_t *problem)
{
    const pr_bcsr_graph_t *graph  = problem->graph;
    const graph_size_t     bcount = graph->bcount;

    for (graph_size_t row = 0; row < bcount; row++)
        for (graph_size_t col = 0; col < bcount; col++)
            if (graph->blocks[(row * bcount) + col]->ecount > 0)
            {
                struct starpu_task *task = starpu_task_pagerank(problem, &cl_pagerank_redux_rank_tmp_pull);

                task->handles[0] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DST][row];
                task->handles[1] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][col];
                task->handles[2] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_RID][(row * bcount) + col];
                task->handles[3] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_CID][(row * bcount) + col];
                task->handles[4] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_SCR][row];

                STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
            }

    starpu_task_invalidate_all(problem, E_PR_PROBLEM_BLOCKS_TMP_VTX);
}

static void starpu_task_pagerank_redux_rank_tmp_push(spu_pr_problem_t *problem)
{
    const pr_bcsr_graph_t *graph  = problem->graph;
    const graph_size_t     bcount = graph->bcount;

    for (graph_size_t row = 0; row < bcount; row++)
        for (graph_size_t col = 0; col < bcount; col++)
            if (graph->blocks[(row * bcount) + col]->ecount > 0)
            {
                struct starpu_task *task = starpu_task_pagerank(problem, &cl_pagerank_redux_rank_tmp_push);

                task->handles[0] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DST][col];
                task->handles[1] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][row];
                task->handles[2] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_RID][(row * bcount) + col];
                task->handles[3] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_CID][(row * bcount) + col];
                task->handles[4] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_SCR][col];

                STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
            }

    starpu_task_invalidate_all(problem, E_PR_PROBLEM_BLOCKS_TMP_VTX);
}

static void starpu_task_pagerank_update_dest(spu_pr_problem_t *problem)
{
    const graph_size_t bcount = problem->graph->bcount;

    for (graph_size_t block = 0; block < bcount; block++)
    {
        struct starpu_task *task = starpu_task_pagerank(problem, &cl_pagerank_update_dest);

        task->handles[0] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block];
        task->handles[1] = problem->data_global[E_PR_PROBLEM_GLOBAL_RNK];

        STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
    }

    starpu_data_invalidate_submit(problem->data_global[E_PR_PROBLEM_GLOBAL_RNK]);
}

static void starpu_task_pagerank_calc_dest(spu_pr_problem_t *problem)
{
    const graph_size_t bcount = problem->graph->bcount;

    for (graph_size_t block = 0; block < bcount; block++)
    {
        struct starpu_task *task = starpu_task_pagerank(problem, &cl_pagerank_calc_dest);

        task->handles[0] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block];
        task->handles[1] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DST][block];
        task->handles[2] = problem->data_global[E_PR_PROBLEM_GLOBAL_RNK];

        STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
    }

    starpu_task_invalidate_all(problem, E_PR_PROBLEM_BLOCKS_TMP_DST);
    starpu_data_invalidate_submit(problem->data_global[E_PR_PROBLEM_GLOBAL_RNK]);
}

static void starpu_task_pagerank_calc_diff(spu_pr_problem_t *problem)
{
    const graph_size_t bcount = problem->graph->bcount;

    for (graph_size_t block = 0; block < bcount; block++)
    {
        struct starpu_task *task = starpu_task_pagerank(problem, &cl_pagerank_calc_diff);

        task->handles[0] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DIF][block];
        task->handles[1] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][block];
        task->handles[2] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][block];

        STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
    }

    starpu_task_redux_sum_all(problem, E_PR_PROBLEM_GLOBAL_DIF, E_PR_PROBLEM_BLOCKS_TMP_DIF);
    starpu_task_invalidate_all(problem, E_PR_PROBLEM_BLOCKS_TMP_DIF);
}

static inline void spu_pr_problem_swap_src_dst(spu_pr_problem_t *problem)
{
    const graph_size_t bcount = problem->graph->bcount;

    for (graph_size_t b = 0; b < bcount; b++)
        SWAP_VALUES(problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][b], problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][b]);
}

static uint32_t spu_pr_problem_run(spu_pr_problem_t *problem)
{
    starpu_task_pagerank_initialize(problem);

    starpu_iteration_push(0);
    problem->iteration_func(problem);
    sem_wait(&problem->semaphore);

    starpu_task_pagerank_finalize(problem);
    starpu_iteration_pop();

    uint32_t iterations = problem->iterations;
    spu_pr_problem_free(problem);

    starpu_pause();
    return iterations;
}

static uint32_t spu_pr_problem_run_bcsc(const pr_bcsc_graph_t *graph, const spu_pr_problem_iteration_func_t iteration_func, pagerank_options_t *options)
{
    starpu_resume();
    return spu_pr_problem_run(spu_pr_problem_new_bcsc(graph, iteration_func, options));
}

static uint32_t spu_pr_problem_run_bcsr(const pr_bcsr_graph_t *graph, const spu_pr_problem_iteration_func_t iteration_func, pagerank_options_t *options)
{
    starpu_resume();
    return spu_pr_problem_run(spu_pr_problem_new_bcsr(graph, iteration_func, options));
}

static void starpu_task_pagerank_mapped_iteration(spu_pr_problem_t *problem)
{
    for (uint32_t local_iterations = 0; local_iterations < problem->options->local_iterations; local_iterations++)
    {
        starpu_iteration_push(problem->iterations + local_iterations);
        spu_pr_problem_swap_src_dst(problem);

        starpu_task_pagerank_baserank(problem);
        starpu_task_pagerank_baserank_redux(problem);

        if (problem->pull)
            starpu_task_pagerank_update_rank_pull(problem);
        else
            starpu_task_pagerank_update_rank_push(problem);
        starpu_task_pagerank_update_dest(problem);
        starpu_iteration_pop();
    }

    starpu_task_pagerank_calc_diff(problem);
    spu_pr_problem_callback(problem);
}

static void starpu_task_pagerank_stepped_iteration(spu_pr_problem_t *problem)
{
    for (uint32_t local_iterations = 0; local_iterations < problem->options->local_iterations; local_iterations++)
    {
        starpu_iteration_push(problem->iterations + local_iterations);
        spu_pr_problem_swap_src_dst(problem);

        starpu_task_pagerank_baserank(problem);
        starpu_task_pagerank_baserank_redux(problem);

        starpu_task_pagerank_update_tmp(problem);
        if (problem->pull)
            starpu_task_pagerank_update_rank_tmp_pull(problem);
        else
            starpu_task_pagerank_update_rank_tmp_push(problem);
        starpu_task_pagerank_update_dest(problem);
        starpu_iteration_pop();
    }

    starpu_task_pagerank_calc_diff(problem);
    spu_pr_problem_callback(problem);
}

static void starpu_task_pagerank_redux_iteration(spu_pr_problem_t *problem)
{
    for (uint32_t local_iterations = 0; local_iterations < problem->options->local_iterations; local_iterations++)
    {
        starpu_iteration_push(problem->iterations + local_iterations);
        spu_pr_problem_swap_src_dst(problem);

        starpu_task_pagerank_baserank(problem);
        starpu_task_pagerank_baserank_redux(problem);

        starpu_task_pagerank_update_tmp(problem);
        if (problem->pull)
            starpu_task_pagerank_redux_rank_tmp_pull(problem);
        else
            starpu_task_pagerank_redux_rank_tmp_push(problem);
        starpu_task_pagerank_calc_dest(problem);
        starpu_iteration_pop();
    }

    starpu_task_pagerank_calc_diff(problem);
    spu_pr_problem_callback(problem);
}

uint32_t pagerank_bcsc_spu_mapped (const pr_bcsc_graph_t *graph, pagerank_options_t *options) { return spu_pr_problem_run_bcsc(graph, starpu_task_pagerank_mapped_iteration,  options); }
uint32_t pagerank_bcsc_spu_stepped(const pr_bcsc_graph_t *graph, pagerank_options_t *options) { return spu_pr_problem_run_bcsc(graph, starpu_task_pagerank_stepped_iteration, options); }
uint32_t pagerank_bcsc_spu_redux  (const pr_bcsc_graph_t *graph, pagerank_options_t *options) { return spu_pr_problem_run_bcsc(graph, starpu_task_pagerank_redux_iteration,   options); }

uint32_t pagerank_bcsr_spu_mapped (const pr_bcsr_graph_t *graph, pagerank_options_t *options) { return spu_pr_problem_run_bcsr(graph, starpu_task_pagerank_mapped_iteration,  options); }
uint32_t pagerank_bcsr_spu_stepped(const pr_bcsr_graph_t *graph, pagerank_options_t *options) { return spu_pr_problem_run_bcsr(graph, starpu_task_pagerank_stepped_iteration, options); }
uint32_t pagerank_bcsr_spu_redux  (const pr_bcsr_graph_t *graph, pagerank_options_t *options) { return spu_pr_problem_run_bcsr(graph, starpu_task_pagerank_redux_iteration,   options); }
