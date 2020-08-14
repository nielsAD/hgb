// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/spu_problem.h"
#include "alg/pagerank/spu_codelets.h"
#include "util/memory.h"

#ifndef STARPU_MAIN_RAM
    #define STARPU_MAIN_RAM 0
#endif
#ifndef STARPU_AUTO_TMP
    #define STARPU_AUTO_TMP -1
#endif

static void spu_pr_problem_register_data(spu_pr_problem_t *problem, const bool deg_in)
{
    assert(problem != NULL);
    assert(problem->graph != NULL);

    const pr_bcsr_graph_t *graph  = problem->graph;
    const graph_size_t     bcount = graph->bcount;

    starpu_variable_data_register(&problem->data_global[E_PR_PROBLEM_GLOBAL_RNK], STARPU_AUTO_TMP, 0, sizeof(*problem->options->result));
    starpu_data_set_reduction_methods(problem->data_global[E_PR_PROBLEM_GLOBAL_RNK], &cl_pagerank_redux_add_single, &cl_pagerank_redux_zero_single);

    starpu_variable_data_register(&problem->data_global[E_PR_PROBLEM_GLOBAL_DIF], STARPU_MAIN_RAM, (uintptr_t) &problem->diff_sum, sizeof(problem->diff_sum));
    starpu_data_set_reduction_methods(problem->data_global[E_PR_PROBLEM_GLOBAL_DIF], &cl_pagerank_redux_add_single, &cl_pagerank_redux_zero_single);
    starpu_data_invalidate(problem->data_global[E_PR_PROBLEM_GLOBAL_DIF]);

    starpu_vector_data_register(&problem->data_global[E_PR_PROBLEM_GLOBAL_SCR], STARPU_AUTO_TMP, 0, PAGERANK_SCRATCH_SIZE, sizeof(*problem->options->result));

    for (graph_size_t b = 0; b < bcount; b++)
    {
        const pr_csr_graph_t *bg = graph->blocks_diag[b];

        starpu_vector_data_register(&problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_SCR][b], STARPU_AUTO_TMP, 0, PAGERANK_SCRATCH_SIZE, sizeof(graph_size_t));
        starpu_vector_data_register(&problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][b], STARPU_AUTO_TMP, 0, bg->vcount, sizeof(*problem->options->result));
        starpu_vector_data_register(&problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_RNK][b], STARPU_AUTO_TMP, 0, bg->vcount, sizeof(*problem->options->result));
        starpu_vector_data_register(&problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DIF][b], STARPU_AUTO_TMP, 0, bg->vcount, sizeof(problem->diff_sum));

        starpu_vector_data_register(&problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DST][b], STARPU_AUTO_TMP, 0, bg->vcount, sizeof(*problem->options->result));
        starpu_data_set_reduction_methods(problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DST][b], &cl_pagerank_redux_add, &cl_pagerank_redux_zero);

        if (deg_in)
            starpu_vector_data_register(&problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][b], STARPU_MAIN_RAM, (uintptr_t) bg->deg_i, bg->vcount, sizeof(*bg->deg_i));
        else
            starpu_vector_data_register(&problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][b], STARPU_MAIN_RAM, (uintptr_t) bg->deg_o, bg->vcount, sizeof(*bg->deg_o));

        starpu_vector_data_register(&problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][b], STARPU_AUTO_TMP, 0, ROUND_TO_MULT(bg->vcount, BCSR_GRAPH_VERTEX_PACK), sizeof(*problem->options->result));
        starpu_vector_data_register(&problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][b], STARPU_AUTO_TMP, 0, ROUND_TO_MULT(bg->vcount, BCSR_GRAPH_VERTEX_PACK), sizeof(*problem->options->result));
    }

    for (graph_size_t b = 0; b < bcount*bcount; b++)
    {
        const pr_csr_graph_t *bg = graph->blocks[b];
        starpu_vector_data_register(&problem->data_blocks[E_PR_PROBLEM_BLOCKS_RID][b], STARPU_MAIN_RAM, (uintptr_t) bg->row_idx, bg->vcount + 1, sizeof(*bg->row_idx));

        if (bg->ecount > 0)
            starpu_vector_data_register(&problem->data_blocks[E_PR_PROBLEM_BLOCKS_CID][b], STARPU_MAIN_RAM, (uintptr_t) bg->col_idx, bg->ecount, sizeof(*bg->col_idx));
    }
}

static void spu_pr_problem_unregister_data(const spu_pr_problem_t *problem)
{
    assert(problem != NULL);
    assert(problem->graph != NULL);

    const graph_size_t bcount = problem->graph->bcount;

    for (size_t handle = 0; handle < E_PR_PROBLEM_GLOBAL_MAX; handle++)
        if (problem->data_global[handle])
            starpu_data_unregister_no_coherency(problem->data_global[handle]);

    for (size_t handle = 0; handle < E_PR_PROBLEM_BLOCKS_MAX; handle++)
        for (graph_size_t b = 0; b < bcount*bcount; b++)
            if (problem->data_blocks[handle][b])
                starpu_data_unregister_no_coherency(problem->data_blocks[handle][b]);
}

spu_pr_problem_t *spu_pr_problem_new(const spu_pr_problem_iteration_func_t iteration_func, const bool pull, pagerank_options_t *options)
{
    spu_pr_problem_t *problem = memory_talloc(spu_pr_problem_t);
    assert(problem != NULL);

    problem->iteration_func = iteration_func;
    problem->options        = options;

    problem->pull       = pull;
    problem->iterations = 0;
    problem->diff_sum   = 1.0;

    sem_init(&problem->semaphore, 0, 0U);
    return problem;
}

spu_pr_problem_t *spu_pr_problem_new_bcsc(const pr_bcsc_graph_t *graph, const spu_pr_problem_iteration_func_t iteration_func, pagerank_options_t *options)
{
    assert(graph != NULL);

    spu_pr_problem_t *problem = spu_pr_problem_new(iteration_func, true, options);
    problem->graph = (pr_bcsc_graph_t*) graph;

    spu_pr_problem_register_data(problem, true);

    return problem;
}

spu_pr_problem_t *spu_pr_problem_new_bcsr(const pr_bcsr_graph_t *graph, const spu_pr_problem_iteration_func_t iteration_func, pagerank_options_t *options)
{
    assert(graph != NULL);

    spu_pr_problem_t *problem = spu_pr_problem_new(iteration_func, false, options);
    problem->graph = (pr_bcsr_graph_t*) graph;

    spu_pr_problem_register_data(problem, false);

    return problem;
}

void spu_pr_problem_free(spu_pr_problem_t *problem)
{
    spu_pr_problem_unregister_data(problem);

    sem_destroy(&problem->semaphore);
    memory_free((void*)problem);
}

void spu_pr_problem_iteration(spu_pr_problem_t *problem)
{
    assert(problem != NULL);

    starpu_data_release(problem->data_global[E_PR_PROBLEM_GLOBAL_DIF]);
    starpu_data_invalidate_submit(problem->data_global[E_PR_PROBLEM_GLOBAL_DIF]);

    problem->iterations += problem->options->local_iterations;

    if ((problem->iterations < problem->options->min_iterations) || (problem->iterations < problem->options->max_iterations && problem->diff_sum > problem->options->epsilon))
    {
        starpu_iteration_push(problem->iterations);
        problem->iteration_func(problem);
        starpu_iteration_pop();
    }
    else
        sem_post(&problem->semaphore);
}

void spu_pr_problem_callback(void *problem)
{
    assert(problem != NULL);

    STARPU_CHECK_RETURN_VALUE(starpu_data_acquire_cb(
        ((spu_pr_problem_t*)problem)->data_global[E_PR_PROBLEM_GLOBAL_DIF],
        STARPU_R,
        (void(*)(void*))spu_pr_problem_iteration,
        problem
    ), "starpu_data_acquire");
}
