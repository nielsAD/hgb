// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/cuda_problem.h"
#include "alg/pagerank/codelets.h"
#include "util/memory.h"
#include "util/math.h"
#include "util/cuda.h"

static void cuda_pr_problem_register_data(cuda_pr_problem_t *problem, const bool deg_in)
{
    assert(problem != NULL);
    assert(problem->graph != NULL);

    const pr_bcsr_graph_t *graph  = problem->graph;
    const graph_size_t     bcount = graph->bcount;

    for (graph_size_t b = 0; b < bcount; b++)
        CUDA_ASSERT(cudaStreamCreate(&problem->streams[b]));

    CUDA_ASSERT(cudaMalloc((void**) &problem->data_global_f[E_PR_PROBLEM_GLOBAL_RNK], sizeof(pr_float) * bcount));
    CUDA_ASSERT(cudaMalloc((void**) &problem->data_global_f[E_PR_PROBLEM_GLOBAL_DIF], sizeof(pr_float) * bcount));

    for (graph_size_t b = 0; b < bcount; b++)
    {
        const pr_csr_graph_t *bg = graph->blocks_diag[b];

        CUDA_ASSERT(cudaMalloc((void**) &problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_TMP_SCR][b], sizeof(graph_size_t) * PAGERANK_SCRATCH_SIZE));
        CUDA_ASSERT(cudaMalloc((void**) &problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_SCR][b], sizeof(pr_float) * PAGERANK_SCRATCH_SIZE));

        if (bg->vcount < 1)
            continue;

        CUDA_ASSERT(cudaMalloc((void**) &problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_VTX][b], sizeof(pr_float) * bg->vcount));
        problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_RNK][b] = problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_VTX][b]; //CUDA_ASSERT(cudaMalloc((void**) &problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_RNK][b], sizeof(pr_float) * bg->vcount));
        problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_DIF][b] = problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_VTX][b]; //CUDA_ASSERT(cudaMalloc((void**) &problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_TMP_DIF][b], sizeof(pr_float) * bg->vcount));

        CUDA_ASSERT(cudaMalloc((void**) &problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_SRC][b], sizeof(pr_float) * ROUND_TO_MULT(bg->vcount, BCSR_GRAPH_VERTEX_PACK)));
        CUDA_ASSERT(cudaMalloc((void**) &problem->data_blocks_f[E_PR_PROBLEM_BLOCKS_DST][b], sizeof(pr_float) * ROUND_TO_MULT(bg->vcount, BCSR_GRAPH_VERTEX_PACK)));

        if (deg_in)
        {
            CUDA_ASSERT(cudaMalloc((void**) &problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_DEG][b], sizeof(*bg->deg_i) * bg->vcount));
            CUDA_ASSERT(cudaMemcpyAsync(problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_DEG][b], bg->deg_i, sizeof(*bg->deg_i) * bg->vcount, cudaMemcpyHostToDevice, problem->streams[b]));
        }
        else
        {
            CUDA_ASSERT(cudaMalloc((void**) &problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_DEG][b], sizeof(*bg->deg_o) * bg->vcount));
            CUDA_ASSERT(cudaMemcpyAsync(problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_DEG][b], bg->deg_o, sizeof(*bg->deg_o) * bg->vcount, cudaMemcpyHostToDevice, problem->streams[b]));
        }
    }

    for (graph_size_t b = 0; b < bcount*bcount; b++)
    {
        const pr_csr_graph_t *bg = graph->blocks[b];

        CUDA_ASSERT(cudaMalloc((void**) &problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_RID][b], sizeof(*bg->row_idx) * (bg->vcount + 1)));
        CUDA_ASSERT(cudaMemcpyAsync(problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_RID][b], bg->row_idx, sizeof(*bg->row_idx) * (bg->vcount + 1), cudaMemcpyHostToDevice, problem->streams[b % bcount]));

        if (bg->ecount < 1)
            continue;

        CUDA_ASSERT(cudaMalloc((void**) &problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_CID][b], sizeof(*bg->col_idx) * bg->ecount));
        CUDA_ASSERT(cudaMemcpyAsync(problem->data_blocks_u[E_PR_PROBLEM_BLOCKS_CID][b], bg->col_idx, sizeof(*bg->col_idx) * bg->ecount, cudaMemcpyHostToDevice, problem->streams[b % bcount]));
    }

    cuda_pr_problem_synchronize(problem);
}

static void cuda_pr_problem_unregister_data(const cuda_pr_problem_t *problem)
{
    assert(problem != NULL);
    assert(problem->graph != NULL);

    const pr_bcsr_graph_t *graph  = problem->graph;
    const graph_size_t     bcount = graph->bcount;

    for (size_t handle = 0; handle < E_PR_PROBLEM_GLOBAL_MAX; handle++)
    {
        CUDA_ASSERT(cudaFree(problem->data_global_f[handle]));
        CUDA_ASSERT(cudaFree(problem->data_global_u[handle]));
    }

    for (size_t handle = 0; handle < E_PR_PROBLEM_BLOCKS_MAX; handle++)
        for (graph_size_t b = 0; b < bcount*bcount; b++)
        {
            if (handle == E_PR_PROBLEM_BLOCKS_TMP_RNK || handle == E_PR_PROBLEM_BLOCKS_TMP_DIF)
            {
                // Skip aliases and NULL pointers
                continue;
            }
            CUDA_ASSERT(cudaFree(problem->data_blocks_f[handle][b]));
            CUDA_ASSERT(cudaFree(problem->data_blocks_u[handle][b]));
        }

    for (graph_size_t b = 0; b < bcount; b++)
        CUDA_ASSERT(cudaStreamDestroy(problem->streams[b]));
}

cuda_pr_problem_t *cuda_pr_problem_new(void)
{
    cuda_pr_problem_t *problem = memory_talloc(cuda_pr_problem_t);
    assert(problem != NULL);

    return problem;
}

cuda_pr_problem_t *cuda_pr_problem_new_bcsc(const pr_bcsc_graph_t *graph)
{
    assert(graph != NULL);

    cuda_pr_problem_t *problem = cuda_pr_problem_new();
    problem->graph = (pr_bcsc_graph_t*) graph;

    cuda_pr_problem_register_data(problem, true);

    return problem;
}

cuda_pr_problem_t *cuda_pr_problem_new_bcsr(const pr_bcsr_graph_t *graph)
{
    assert(graph != NULL);

    cuda_pr_problem_t *problem = cuda_pr_problem_new();
    problem->graph = (pr_bcsr_graph_t*) graph;

    cuda_pr_problem_register_data(problem, false);

    return problem;
}

void cuda_pr_problem_free(cuda_pr_problem_t *problem)
{
    cuda_pr_problem_unregister_data(problem);

    memory_free((void*)problem);
}

void cuda_pr_problem_synchronize(const cuda_pr_problem_t *problem)
{
    assert(problem != NULL);
    assert(problem->graph != NULL);

    for (graph_size_t block = 0; block < problem->graph->bcount; block++)
        CUDA_ASSERT(cudaStreamSynchronize(problem->streams[block]));
}
