// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/pcsr.h"
#include "util/mpi.h"

#ifndef GRAPH_NAME
    #include "graph/template/pcsr.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

graph_size_t pcsr_graph(_get_global_vcount)(const pcsr_graph_t *restrict graph, MPI_Comm comm, const bool with_ghost);
graph_size_t pcsr_graph(_get_global_ecount)(const pcsr_graph_t *restrict graph, MPI_Comm comm, const bool with_cross);

void pcsr_graph(_sync_degrees_out)(pcsr_graph_t *restrict graph, MPI_Comm comm);
void pcsr_graph(_sync_degrees_in)(pcsr_graph_t *restrict graph, MPI_Comm comm);

#ifdef __cplusplus
}
#endif

#define pcsr_graph_sync_fan_in_ex(graph, vidx, arr, dt, sbuf, rbuf) { \
    const graph_size_t part = graph->part; \
    const graph_size_t pcount = graph->pcount; \
    int sendcounts[pcount]; int sdispls[pcount]; \
    int recvcounts[pcount]; int rdispls[pcount]; \
    csr_forall_vertices(p, pcount) \
    { \
        const graph_size_t sendcount = (p > 0 && p-1 != part) ? pcsr_cross_graph(_get_vertex_degree_out)(graph->cross_graph, p-1) : 0; \
        sdispls[p] = (p > 0) ? sdispls[p-1] + sendcount : 0; \
        rdispls[p] = (p > 0) ? rdispls[p-1] + recvcounts[p-1] : 0; \
        sendcounts[p] = 0; \
        recvcounts[p] = (p == part) ? 0 : pcsr_cross_graph(_get_vertex_degree_in)(graph->cross_graph, p); \
    } \
    csr_forall_edges(eid, pfr, pto, graph->cross_graph) \
    { \
        if (pfr == part) continue; \
        const graph_tag_t idx = pcsr_cross_graph(_get_edge_tag)(graph->cross_graph, eid); \
        graph_size_t bid, bidx; \
        UNUSED bool res = vidx(graph->local_graph, idx, &bid, &bidx); assert(res); \
        sbuf[sdispls[pfr] + sendcounts[pfr]++] = arr[bid][bidx]; \
    } \
    MPI_Alltoallv( \
        sbuf, sendcounts, sdispls, dt, \
        rbuf, recvcounts, rdispls, dt, \
        comm \
    ); \
    csr_forall_out_edges(eid, pto, part, graph->cross_graph) \
    { \
        const graph_tag_t idx = pcsr_cross_graph(_get_edge_tag)(graph->cross_graph, eid); \
        graph_size_t bid, bidx; \
        UNUSED bool res = vidx(graph->local_graph, idx, &bid, &bidx); assert(res); \
        arr[bid][bidx] += rbuf[rdispls[pto]++]; \
    } \
}

#define pcsr_graph_sync_fan_out_ex(graph, vidx, arr, dt, sbuf, rbuf) { \
    const graph_size_t part = graph->part; \
    const graph_size_t pcount = graph->pcount; \
    int sendcounts[pcount]; int sdispls[pcount]; \
    int recvcounts[pcount]; int rdispls[pcount]; \
    csr_forall_vertices(p, pcount) \
    { \
        const graph_size_t sendcount = (p > 0 && p-1 != part) ? pcsr_cross_graph(_get_vertex_degree_in)(graph->cross_graph, p-1) : 0; \
        sdispls[p] = (p > 0) ? sdispls[p-1] + sendcount : 0; \
        rdispls[p] = (p > 0) ? rdispls[p-1] + recvcounts[p-1] : 0; \
        sendcounts[p] = 0; \
        recvcounts[p] = (p == part) ? 0 : pcsr_cross_graph(_get_vertex_degree_out)(graph->cross_graph, p); \
    } \
    csr_forall_out_edges(eid, pto, part, graph->cross_graph) \
    { \
        const graph_tag_t idx = pcsr_cross_graph(_get_edge_tag)(graph->cross_graph, eid); \
        graph_size_t bid, bidx; \
        UNUSED bool res = vidx(graph->local_graph, idx, &bid, &bidx); assert(res); \
        sbuf[sdispls[pto] + sendcounts[pto]++] = arr[bid][bidx]; \
    } \
    MPI_Alltoallv( \
        sbuf, sendcounts, sdispls, dt, \
        rbuf, recvcounts, rdispls, dt, \
        comm \
    ); \
    csr_forall_edges(eid, pfr, pto, graph->cross_graph) \
    { \
        if (pfr == part) continue; \
        const graph_tag_t idx = pcsr_cross_graph(_get_edge_tag)(graph->cross_graph, eid); \
        graph_size_t bid, bidx; \
        UNUSED bool res = vidx(graph->local_graph, idx, &bid, &bidx); assert(res); \
        arr[bid][bidx] = rbuf[rdispls[pfr]++]; \
    } \
}

#define pcsr_graph_sync_fan_in(graph, vidx, arr, dt) { \
    DECLARE_TYPE_OF(*arr) sbuf = memory_talloc(**arr, pcsr_cross_graph(_get_vertex_degree_in)(graph->cross_graph, graph->part)); \
    DECLARE_TYPE_OF(*arr) rbuf = memory_talloc(**arr, pcsr_cross_graph(_get_vertex_degree_out)(graph->cross_graph, graph->part)); \
    pcsr_graph_sync_fan_in_ex(graph, vidx, arr, dt, sbuf, rbuf) \
    memory_free((void*)sbuf); \
    memory_free((void*)rbuf); \
}

#define pcsr_graph_sync_fan_out(graph, vidx, arr, dt) { \
    DECLARE_TYPE_OF(*arr) sbuf = memory_talloc(**arr, pcsr_cross_graph(_get_vertex_degree_out)(graph->cross_graph, graph->part)); \
    DECLARE_TYPE_OF(*arr) rbuf = memory_talloc(**arr, pcsr_cross_graph(_get_vertex_degree_in)(graph->cross_graph, graph->part)); \
    pcsr_graph_sync_fan_out_ex(graph, vidx, arr, dt, sbuf, rbuf) \
    memory_free((void*)sbuf); \
    memory_free((void*)rbuf); \
}
