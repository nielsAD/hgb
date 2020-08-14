// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "util/igraph.h"

igraph_eli_graph_t *igraph_eli_graph(_from_igraph)(const igraph_t *graph, const graph_flags_enum_t flags)
{
    assert(graph != NULL);
    igraph_eli_graph_t *res = igraph_eli_graph(_new)((graph_flags_enum_t) (flags & ~E_GRAPH_FLAG_SORT));
    if (res == NULL)
        return res;

    res->ecount = igraph_ecount(graph);
    igraph_eli_graph(_grow)(res, res->ecount);

    igraph_eit_t it;
    igraph_eit_create(graph, igraph_ess_all(IGRAPH_EDGEORDER_ID), &it);

    for (IGRAPH_EIT_RESET(it); !IGRAPH_EIT_END(it); IGRAPH_EIT_NEXT(it))
    {
        const igraph_integer_t edge = IGRAPH_EIT_GET(it);
        const igraph_integer_t from = IGRAPH_FROM(graph, edge);
        const igraph_integer_t to   = IGRAPH_TO(graph, edge);

        res->efr[edge] = from;
        res->eto[edge] = to;
    }

    igraph_eit_destroy(&it);

    igraph_eli_graph(_toggle_flag)(res, E_GRAPH_FLAG_SORT, E_GRAPH_FLAG_SORT & flags);
    return res;
}

igraph_csr_graph_t *igraph_csr_graph(_from_igraph)(const igraph_t *graph, const graph_flags_enum_t flags)
{
    igraph_eli_graph_t *eli = igraph_eli_graph(_from_igraph)(graph, flags);
    igraph_csr_graph_t *res = igraph_csr_graph(_convert_from_eli)(eli, flags);
    igraph_eli_graph(_free)(eli);

    return res;
}
