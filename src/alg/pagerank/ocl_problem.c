// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/ocl_problem.h"
#include "alg/pagerank/codelets.h"
#include "util/memory.h"
#include "util/math.h"

static void ocl_pr_problem_register_data(ocl_pr_problem_t *problem, const bool deg_in)
{
    assert(problem != NULL);
    assert(problem->graph != NULL);

    cl_int err;
    const pr_bcsr_graph_t  *graph   = problem->graph;
    const graph_size_t      bcount  = graph->bcount;
    const cl_context        context = problem->context;
    const cl_command_queue  queue   = problem->queue;

    cl_event events[bcount + 2*bcount*bcount];
    size_t   event_idx = 0;

    problem->data_global[E_PR_PROBLEM_GLOBAL_RNK] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(pr_float) * bcount, NULL, &err); OPENCL_ASSERT(err);
    problem->data_global[E_PR_PROBLEM_GLOBAL_DIF] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(pr_float) * bcount, NULL, &err); OPENCL_ASSERT(err);

    for (graph_size_t b = 0; b < bcount; b++)
    {
        const pr_csr_graph_t *bg = graph->blocks_diag[b];

        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_SCR][b] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(pr_float) * PAGERANK_SCRATCH_SIZE, NULL, &err); OPENCL_ASSERT(err);

        if (bg->vcount < 1)
            continue;

        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][b] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(pr_float) * bg->vcount, NULL, &err); OPENCL_ASSERT(err);
        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_RNK][b] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][b]; //clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(pr_float) * bg->vcount, NULL, &err); OPENCL_ASSERT(err);
        problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_DIF][b] = problem->data_blocks[E_PR_PROBLEM_BLOCKS_TMP_VTX][b]; //clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(pr_float) * bg->vcount, NULL, &err); OPENCL_ASSERT(err);

        problem->data_blocks[E_PR_PROBLEM_BLOCKS_SRC][b] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(pr_float) * ROUND_TO_MULT(bg->vcount, BCSR_GRAPH_VERTEX_PACK), NULL, &err); OPENCL_ASSERT(err);
        problem->data_blocks[E_PR_PROBLEM_BLOCKS_DST][b] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(pr_float) * ROUND_TO_MULT(bg->vcount, BCSR_GRAPH_VERTEX_PACK), NULL, &err); OPENCL_ASSERT(err);

        if (deg_in)
        {
            problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][b] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(*bg->deg_i) * bg->vcount, NULL, &err); OPENCL_ASSERT(err);
            OPENCL_ASSERT(clEnqueueWriteBuffer(queue, problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][b], CL_FALSE, 0, sizeof(*bg->deg_i) * bg->vcount, bg->deg_i, 0, NULL, &events[event_idx++]));
        }
        else
        {
            problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][b] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(*bg->deg_o) * bg->vcount, NULL, &err); OPENCL_ASSERT(err);
            OPENCL_ASSERT(clEnqueueWriteBuffer(queue, problem->data_blocks[E_PR_PROBLEM_BLOCKS_DEG][b], CL_FALSE, 0, sizeof(*bg->deg_o) * bg->vcount, bg->deg_o, 0, NULL, &events[event_idx++]));
        }
    }

    for (graph_size_t b = 0; b < bcount*bcount; b++)
    {
        const pr_csr_graph_t *bg = graph->blocks[b];

        problem->data_blocks[E_PR_PROBLEM_BLOCKS_RID][b] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(*bg->row_idx) * (bg->vcount + 1), NULL, &err); OPENCL_ASSERT(err);
        OPENCL_ASSERT(clEnqueueWriteBuffer(queue, problem->data_blocks[E_PR_PROBLEM_BLOCKS_RID][b], CL_FALSE, 0, sizeof(*bg->row_idx) * (bg->vcount + 1), bg->row_idx, 0, NULL, &events[event_idx++]));

        if (bg->ecount < 1)
            continue;

        problem->data_blocks[E_PR_PROBLEM_BLOCKS_CID][b] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(*bg->col_idx) * bg->ecount, NULL, &err); OPENCL_ASSERT(err);
        OPENCL_ASSERT(clEnqueueWriteBuffer(queue, problem->data_blocks[E_PR_PROBLEM_BLOCKS_CID][b], CL_FALSE, 0, sizeof(*bg->col_idx) * bg->ecount, bg->col_idx, 0, NULL, &events[event_idx++]));
    }

    OPENCL_ASSERT(clWaitForEvents(event_idx, events));

    for (size_t i = 0; i < event_idx; i++)
        clReleaseEvent(events[i]);
}

static void ocl_pr_problem_unregister_data(const ocl_pr_problem_t *problem)
{
    assert(problem != NULL);
    assert(problem->graph != NULL);

    const pr_bcsr_graph_t *graph  = problem->graph;
    const graph_size_t     bcount = graph->bcount;

    for (size_t handle = 0; handle < E_PR_PROBLEM_GLOBAL_MAX; handle++)
        if (problem->data_global[handle])
            OPENCL_ASSERT(clReleaseMemObject(problem->data_global[handle]));

    for (size_t handle = 0; handle < E_PR_PROBLEM_BLOCKS_MAX; handle++)
        for (graph_size_t b = 0; b < bcount*bcount; b++)
        {
            if (!problem->data_blocks[handle][b] || handle == E_PR_PROBLEM_BLOCKS_TMP_RNK || handle == E_PR_PROBLEM_BLOCKS_TMP_DIF)
            {
                // Skip aliases and NULL pointers
                continue;
            }
            OPENCL_ASSERT(clReleaseMemObject(problem->data_blocks[handle][b]));
        }
}

ocl_pr_problem_t *ocl_pr_problem_new(const int devid)
{
    ocl_pr_problem_t *problem = memory_talloc(ocl_pr_problem_t);
    assert(problem != NULL);

    starpu_opencl_get_context(devid, &problem->context);
    starpu_opencl_get_queue(devid, &problem->queue);

    return problem;
}

ocl_pr_problem_t *ocl_pr_problem_new_bcsc(const int devid, const pr_bcsc_graph_t *graph)
{
    assert(graph != NULL);

    ocl_pr_problem_t *problem = ocl_pr_problem_new(devid);
    problem->graph = (pr_bcsc_graph_t*) graph;

    ocl_pr_problem_register_data(problem, true);

    return problem;
}

ocl_pr_problem_t *ocl_pr_problem_new_bcsr(const int devid, const pr_bcsr_graph_t *graph)
{
    assert(graph != NULL);

    ocl_pr_problem_t *problem = ocl_pr_problem_new(devid);
    problem->graph = (pr_bcsr_graph_t*) graph;

    ocl_pr_problem_register_data(problem, false);

    return problem;
}

void ocl_pr_problem_free(ocl_pr_problem_t *problem)
{
    assert(problem != NULL);

    ocl_pr_problem_unregister_data(problem);
    ocl_pr_problem_clear_events(problem);

    memory_free((void*)problem);
}

void ocl_pr_problem_clear_events(ocl_pr_problem_t *problem)
{
    for (size_t handle = 0; handle < E_PR_PROBLEM_GLOBAL_MAX; handle++)
        for (graph_size_t block = 0; block < BCSR_GRAPH_MAX_BCOUNT*BCSR_GRAPH_MAX_BCOUNT; block++)
            ocl_pr_problem_set_event_g(problem, (pagerank_problem_data_global_enum_t) handle, block, NULL);

    for (size_t handle = 0; handle < E_PR_PROBLEM_BLOCKS_MAX; handle++)
        for (graph_size_t block = 0; block < BCSR_GRAPH_MAX_BCOUNT*BCSR_GRAPH_MAX_BCOUNT; block++)
            ocl_pr_problem_set_event_b(problem, (pagerank_problem_data_block_enum_t) handle, block, NULL);
}

cl_event ocl_pr_problem_set_event_g(ocl_pr_problem_t *problem, const pagerank_problem_data_global_enum_t index, const graph_size_t block, const cl_event event)
{
    assert(problem != NULL);
    assert(index < E_PR_PROBLEM_GLOBAL_MAX);
    assert(block < BCSR_GRAPH_MAX_BCOUNT*BCSR_GRAPH_MAX_BCOUNT);

    cl_event prev = problem->event_global[index][block];
    if (prev) clReleaseEvent(prev);

    problem->event_global[index][block] = event;

    return prev;
}

cl_event ocl_pr_problem_set_event_b(ocl_pr_problem_t *problem, const pagerank_problem_data_block_enum_t index, const graph_size_t block, const cl_event event)
{
    assert(problem != NULL);
    assert(index < E_PR_PROBLEM_BLOCKS_MAX);
    assert(block < BCSR_GRAPH_MAX_BCOUNT*BCSR_GRAPH_MAX_BCOUNT);

    cl_event prev = problem->event_blocks[index][block];
    if (prev) clReleaseEvent(prev);

    problem->event_blocks[index][block] = event;

    return prev;
}
