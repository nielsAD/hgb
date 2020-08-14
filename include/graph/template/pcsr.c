// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/pcsr.h"
#include "util/math.h"
#include "util/memory.h"

#ifndef GRAPH_NAME
    #include "graph/template/pcsr.h"
#endif

static bool _pcsr_cross_filter_mapping_fun_e(const graph_size_t old_src, const graph_size_t old_dst, graph_size_t *restrict new_src, graph_size_t *restrict new_dst, graph_size_t *restrict part);

pcsr_graph_t *pcsr_graph(_new_ex)(const graph_size_t part, const graph_size_t pcount, const graph_size_t vcount, pcsr_local_graph_t *restrict local, pcsr_cross_graph_t *restrict cross)
{
    assert(pcount <= PCSR_GRAPH_MAX_PCOUNT);
    assert(part < pcount);

    pcsr_graph_t *graph = memory_talloc(pcsr_graph_t);
    assert(graph != NULL);

    graph->local_graph = (local != NULL) ? local : pcsr_local_graph(_new)(1, E_GRAPH_FLAG_NONE);
    graph->cross_graph = (cross != NULL) ? cross : pcsr_cross_graph(_new)(PCSR_CROSS_FLAGS_INHERIT(graph->local_graph->flags));

    if (graph->cross_graph->vcount < pcount)
        pcsr_cross_graph(_add_vertices)(
            graph->cross_graph,
            pcount - graph->cross_graph->vcount
        );

    graph_size_t max_v = 0;
    csr_forall_edges_par(eid, src, dst, graph->cross_graph, reduction(max:max_v))
    {
        if (src != part && dst != part) continue;
        const graph_size_t idx = pcsr_cross_graph(_get_edge_tag)(graph->cross_graph, eid);
        if (idx > max_v) max_v = idx;
    }

    if (max_v >= graph->local_graph->vcount)
        pcsr_local_graph(_add_vertices)(graph->local_graph, max_v - graph->local_graph->vcount + 1);

    graph->part   = part;
    graph->pcount = pcount;
    graph->vcount = vcount;

    return graph;
}

pcsr_graph_t *pcsr_graph(_new)(const graph_size_t part, const graph_size_t pcount, const graph_size_t vcount, const graph_size_t bcount, const graph_flags_enum_t flags)
{
    return pcsr_graph(_new_ex)(
        part, pcount, vcount,
        pcsr_local_graph(_new)(bcount, flags),
        pcsr_cross_graph(_new)(PCSR_CROSS_FLAGS_INHERIT(flags))
    );
}

pcsr_graph_t *pcsr_graph(_copy)(const pcsr_graph_t *base, graph_flags_enum_t flags)
{
    assert(base != NULL);

    return pcsr_graph(_new_ex)(
        base->part, base->pcount, base->vcount,
        pcsr_local_graph(_copy)(base->local_graph, flags),
        pcsr_cross_graph(_copy)(base->cross_graph, GRAPH_VALID_FLAGS_MASK)
    );
}

pcsr_graph_t *pcsr_graph(_read_file_transposed)(const graph_size_t part, const graph_size_t pcount, const graph_size_t bcount, const char *const base, const bool transposed_local, const bool transposed_cross, const graph_flags_enum_t flags, const char *const force_ext)
{
    if (pcount <= 1)
    {
        pcsr_local_graph_t *local = pcsr_local_graph(_read_file)(base, flags, bcount, force_ext);
        return pcsr_graph(_new_ex)(part, pcount, local->vcount, local, NULL);
    }

    char *localname = pcsr_graph_filename_partition(base, pcount, part);

    pcsr_local_graph_t *local = pcsr_local_graph(_read_file)(localname, flags, bcount, force_ext);
    pcsr_cross_graph_t *cross = pcsr_cross_graph_read_file(part, pcount, transposed_cross, flags, base);
    const graph_size_t vcount = pcsr_global_graph_vcount(pcount, base);

    free(localname);

    if (local != NULL && transposed_local)
        pcsr_local_graph(_transpose)(local);

    if (local == NULL || cross == NULL)
    {
        if (local != NULL) pcsr_local_graph(_free)(local);
        if (cross != NULL) pcsr_cross_graph(_free)(cross);
        return NULL;
    }

    return pcsr_graph(_new_ex)(part, pcount, vcount, local, cross);
}

pcsr_graph_t *pcsr_graph(_read_file)(const graph_size_t part, const graph_size_t pcount, const graph_size_t bcount, const char *const base, const graph_flags_enum_t flags, const char *const force_ext)
{
    return pcsr_graph(_read_file_transposed)(part, pcount, bcount, base, false, false, flags, force_ext);
}

void pcsr_graph(_free)(pcsr_graph_t *graph)
{
    assert(graph != NULL);

    pcsr_cross_graph(_free)(graph->cross_graph);
    pcsr_local_graph(_free)(graph->local_graph);
    memory_free((void*)graph);
}

bool pcsr_graph(_equals)(const pcsr_graph_t *restrict first, const pcsr_graph_t *restrict second)
{
    if (first == second)
        return true;

    assert(first != NULL && second != NULL);

    return first->part == second->part && first->pcount == second->pcount
        && pcsr_cross_graph(_equals)(first->cross_graph, second->cross_graph)
        && pcsr_local_graph(_equals)(first->local_graph, second->local_graph);
}

void pcsr_graph(_filter_cross_graph)(pcsr_graph_t *restrict graph)
{
    assert(graph != NULL);
    pcsr_cross_graph_t *cross = graph->cross_graph;
    graph->cross_graph = pcsr_cross_graph(_mapped_copy)(
        cross,
        (graph_flags_enum_t) (cross->flags & ~E_GRAPH_FLAG_SORT),
        graph_map_vertex_noop,
        (graph_map_edge_func_t)_pcsr_cross_filter_mapping_fun_e,
        &graph->part
    );
    pcsr_cross_graph(_free)(cross);
}
