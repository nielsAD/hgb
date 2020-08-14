// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/pcsr_mpi.h"
#include "util/memory.h"

#ifndef GRAPH_NAME
    #include "graph/template/pcsr_mpi.h"
#endif

graph_size_t pcsr_graph(_get_global_vcount)(const pcsr_graph_t *restrict graph, MPI_Comm comm, const bool with_ghost)
{
    assert(graph != NULL);
    assert(comm != MPI_COMM_NULL);
    graph_size_t vcount = graph->local_graph->vcount -
        (with_ghost ? 0 : pcsr_cross_graph(_get_vertex_degree_in)(graph->cross_graph, graph->part));
    MPI_Allreduce(&vcount, &vcount, 1, mpi_get_int_type(graph_size_t), MPI_SUM, comm);
    return vcount;
}

graph_size_t pcsr_graph(_get_global_ecount)(const pcsr_graph_t *restrict graph, MPI_Comm comm, const bool with_cross)
{
    assert(graph != NULL);
    assert(comm != MPI_COMM_NULL);
    graph_size_t ecount = graph->local_graph->ecount +
        (with_cross ? pcsr_cross_graph(_get_vertex_degree_out)(graph->cross_graph, graph->part) : 0);
    MPI_Allreduce(&ecount, &ecount, 1, mpi_get_int_type(graph_size_t), MPI_SUM, comm);
    return ecount;
}

void pcsr_graph(_sync_degrees_out)(pcsr_graph_t *restrict graph, MPI_Comm comm)
{
    assert(graph != NULL);
    assert(comm != MPI_COMM_NULL);
    assert(graph->local_graph->flags & E_GRAPH_FLAG_DEG_O);

    graph_size_t *restrict arr[graph->local_graph->bcount];
    bcsr_forall_diag_blocks(bid, graph->local_graph)
    {
        arr[bid] = graph->local_graph->blocks_diag[bid]->deg_o;
    }

    const MPI_Datatype MPI_GRAPH_SIZE_T = mpi_get_int_type(graph_size_t);
    pcsr_graph_sync_fan_out(graph, pcsr_local_graph(_get_vertex_index), arr, MPI_GRAPH_SIZE_T)
}

void pcsr_graph(_sync_degrees_in)(pcsr_graph_t *restrict graph, MPI_Comm comm)
{
    assert(graph != NULL);
    assert(comm != MPI_COMM_NULL);
    assert(graph->local_graph->flags & E_GRAPH_FLAG_DEG_I);

    graph_size_t *restrict sbuf = memory_talloc(graph_size_t, pcsr_cross_graph(_get_vertex_degree_out)(graph->cross_graph, graph->part));
    graph_size_t *restrict rbuf = memory_talloc(graph_size_t, pcsr_cross_graph(_get_vertex_degree_in)(graph->cross_graph, graph->part));

    graph_size_t *restrict arr[graph->local_graph->bcount];
    bcsr_forall_diag_blocks(bid, graph->local_graph)
    {
        arr[bid] = graph->local_graph->blocks_diag[bid]->deg_i;
    }

    const MPI_Datatype MPI_GRAPH_SIZE_T = mpi_get_int_type(graph_size_t);
    pcsr_graph_sync_fan_in_ex(graph, pcsr_local_graph(_get_vertex_index), arr, MPI_GRAPH_SIZE_T, rbuf, sbuf)
    pcsr_graph_sync_fan_out_ex(graph, pcsr_local_graph(_get_vertex_index), arr, MPI_GRAPH_SIZE_T, sbuf, rbuf)

    memory_free((void*)sbuf);
    memory_free((void*)rbuf);
}
