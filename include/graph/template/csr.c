// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/csr.h"
#include "util/math.h"
#include "util/memory.h"

#ifndef GRAPH_NAME
    #include "graph/template/csr.h"
#endif

static int _reorder_col_idx_compare(const void *a, const void *b);
static int _reorder_by_degree_compare(const void *a, const void *b, void *c);

static bool _idx_mapping_fun_v(const graph_size_t old_index, graph_size_t *restrict new_index, graph_size_t *restrict args);
static bool _idx_mapping_fun_e(const graph_size_t old_src, const graph_size_t old_dst, graph_size_t *restrict new_src, graph_size_t *restrict new_dst, graph_size_t *restrict args);

csr_graph_t *csr_graph(_new_ex)(graph_size_t *restrict rid, graph_size_t *restrict cid, const graph_size_t vcount, const graph_flags_enum_t flags)
{
    assert(!(~CSR_VALID_FLAGS_MASK & flags));

    #ifndef GRAPH_V_TYPE
        assert(!(E_GRAPH_FLAG_VAL_V & flags));
    #endif
    #ifndef GRAPH_E_TYPE
        assert(!(E_GRAPH_FLAG_VAL_E & flags));
    #endif

    csr_graph_t *graph = memory_talloc(csr_graph_t);
    assert(graph != NULL);

    graph->row_idx = (rid != NULL)
        ? rid
        : memory_pinned_talloc_if(E_GRAPH_FLAG_PIN & flags, *graph->row_idx);

    graph->col_idx = cid;
    graph->vcount  = vcount;
    graph->ecount  = graph->row_idx[vcount];
    graph->flags   = (graph_flags_enum_t) (flags | E_GRAPH_FLAG_SORT);

    return graph;
}

csr_graph_t *csr_graph(_new)(const graph_flags_enum_t flags)
{
    return csr_graph(_new_ex)(NULL, NULL, 0, flags);
}

csr_graph_t *csr_graph(_new_random)(const graph_size_t vcount, graph_size_t ecount, const graph_flags_enum_t flags)
{
    csr_graph_t *graph = csr_graph(_new)(flags);

    const unsigned long long int max_vcount = (unsigned long long int)vcount * vcount;
    if (ecount > max_vcount)
        ecount = max_vcount;

    csr_graph(_grow)(graph, vcount, ecount);
    csr_graph(_add_vertices)(graph, vcount);

    if (ecount < 1)
        return graph;

    const long double edge_p = ecount / (long double) max_vcount;
    const long double lp = logl(1.0 - edge_p);

    graph_size_t edge_frm = 0;
    graph_size_t edge_to  = 0;
    graph_size_t edge_idx = 0;

    graph->row_idx[0] = 0;

    while (true)
    {
        const long double lr = logl(1.0 - (rand() / (long double) RAND_MAX));
        edge_to += 1 + (int)(lr / lp);

        while ((edge_to >= vcount) && (edge_frm < vcount))
        {
            edge_frm++;
            graph->row_idx[edge_frm] = edge_idx;
            edge_to -= vcount;
        }

        if (edge_frm >= vcount)
            break;
        else if (edge_idx < ecount)
        {
            graph->col_idx[edge_idx] = edge_to;
            edge_idx++;

            if (graph->deg_i != NULL)
                graph->deg_i[edge_to]++;
            if (graph->deg_o != NULL)
                graph->deg_o[edge_frm]++;
        }
    }

    graph->ecount = edge_idx;

    if (graph->etag != NULL)
        memset(graph->etag, 0, sizeof(*graph->etag) * ecount);
    #ifdef GRAPH_E_TYPE
        if (graph->eval != NULL)
            memset(graph->eval, 0, sizeof(*graph->eval) * ecount);
    #endif

    return graph;
}

csr_graph_t *csr_graph(_new_regular)(const graph_size_t vcount, graph_size_t deg, graph_size_t stride, const graph_flags_enum_t flags)
{
    csr_graph_t *graph = csr_graph(_new)(flags);

    csr_graph(_grow)(graph, vcount, vcount * deg);
    csr_graph(_add_vertices)(graph, vcount);

    if (deg < 1)
        return graph;

    csr_forall_vertices_par(v, graph, /*no omp params*/)
    {
        const graph_size_t start = v * deg;
        graph->row_idx[v] = start;

        for (graph_size_t d = 0; d < deg; d++)
        {
            graph->col_idx[start + d] = ((v * stride) + (d * stride)) % vcount;
        }

        if ((v * stride) + ((deg - 1) * stride) >= vcount) {
            qsort(&graph->col_idx[start], deg, sizeof(graph_size_t), _reorder_col_idx_compare);
        }

        if (graph->deg_i != NULL)
            graph->deg_i[v] = deg;
        if (graph->deg_o != NULL)
            graph->deg_o[v] = deg;
    }

    const graph_size_t ecount = vcount * deg;
    graph->ecount = ecount;
    graph->row_idx[vcount] = ecount;

    if (graph->etag != NULL)
        memset(graph->etag, 0, sizeof(*graph->etag) * ecount);
    #ifdef GRAPH_E_TYPE
        if (graph->eval != NULL)
            memset(graph->eval, 0, sizeof(*graph->eval) * ecount);
    #endif

    return graph;
}

csr_graph_t *csr_graph(_new_kronecker)(const graph_size_t vcount, graph_size_t ecount, const graph_flags_enum_t flags)
{
    eli_graph_t *eli = eli_graph(_new_kronecker)(vcount, ecount, (graph_flags_enum_t) (flags & ELI_VALID_FLAGS_MASK));
    csr_graph_t *csr = csr_graph(_convert_from_eli)(eli, flags);
    eli_graph(_free)(eli);

    return csr;
}

csr_graph_t *csr_graph(_read)(FILE *stream, const graph_flags_enum_t flags)
{
    assert(stream != NULL);
    assert(!(E_GRAPH_FLAG_VAL_V & flags));
    assert(!(E_GRAPH_FLAG_VAL_E & flags));

    csr_graph_t *graph = csr_graph(_new)(flags);

    char  *buf_txt = NULL;
    size_t buf_len = 0;
    ssize_t line_len;

    size_t vcount = 0;
    size_t ecount = 0;
    size_t format = 0;
    size_t n_vertex_weight = 1;

    if (getline(&buf_txt, &buf_len, stream) < 3
      ||(sscanf(buf_txt, "%zu %zu %zu %zu",
            &vcount,
            &ecount,
            &format,
            &n_vertex_weight
        ) < 2))
    {
        // Invalid input
        csr_graph(_free)(graph);
        graph = NULL;
    }
    else
    {
        const bool has_vertex_size   = format >= 100; if (has_vertex_size)   format -= 100;
        const bool has_edge_weight   = format >= 10;  if (has_edge_weight)   format -=  10;
        const bool has_vertex_weight = format >= 1;   if (has_vertex_weight) format -=   1;

        assert(format == 0);
        if (!has_vertex_weight)
            n_vertex_weight = 0;

        csr_graph(_grow)(graph, (graph_size_t)vcount, (graph_size_t)ecount);
        csr_graph(_add_vertices)(graph, (graph_size_t)vcount);

        graph_size_t src = 0;
        while ((line_len = getline(&buf_txt, &buf_len, stream)) != -1)
        {
            if (line_len < 1 || buf_txt[0] == '#' || buf_txt[0] == '%')
                continue;

            graph->vcount = src + 1;
            graph->row_idx[src + 1] = graph->row_idx[src];

            char *buf_pos = buf_txt;
            if (has_vertex_size)
                strtoull(buf_pos, &buf_pos, 0); // TODO: read into vval

            if (has_vertex_weight)
            {
                const graph_tag_t w = strtoull(buf_pos, &buf_pos, 0);
                if (E_GRAPH_FLAG_TAG_V & flags)
                    graph->vtag[src] = w;
            }

            // Skip rest of the weights
            for (size_t w = 1; w < n_vertex_weight; w++)
                strtoull(buf_pos, &buf_pos, 0);

            char *buf_end = buf_pos;
            while (true)
            {
                const graph_size_t dst = strtoull(buf_pos, &buf_end, 0) - 1;
                if (buf_pos == buf_end)
                    break;

                assert(dst < (graph_size_t)vcount);

                buf_pos = buf_end;
                const graph_size_t edge = csr_graph(_add_edge)(graph, src, dst);

                if (has_edge_weight)
                {
                    const graph_tag_t w = strtoull(buf_pos, &buf_pos, 0);
                    if (E_GRAPH_FLAG_TAG_E & flags)
                        graph->etag[edge] = w;
                }
            }

            src++;
        }
    }

    free(buf_txt);

    if (graph != NULL)
        csr_graph(_shrink)(graph);

    return graph;
}

csr_graph_t *csr_graph(_read_file)(const char *const filename, const graph_flags_enum_t flags, const char *const force_ext)
{
    const char *ext = (force_ext != NULL) ? force_ext : file_extension(filename);
    if (!ext)
        return NULL;

    csr_graph_t *csr = NULL;
    eli_graph_t *eli = NULL;

    if (strcasecmp(ext, "dimacs") == 0 || strcasecmp(ext, "csr") == 0 || strcasecmp(ext, "csc") == 0)
    {
        FILE* file = (filename == NULL || strcmp(filename, "-") == 0)
            ? stdin
            : fopen(filename, "r");

        if (file)
        {
            csr = csr_graph(_read)(file, flags);
            if (file != stdin)
                fclose(file);

            if (csr != NULL && strcasecmp(ext, "csc") == 0)
            {
                csr_graph(_transpose)(csr);
            }
        }
    }
    else if ((eli = eli_graph(_read_file)(filename, (graph_flags_enum_t) (flags & ELI_VALID_FLAGS_MASK), force_ext)) != NULL)
    {
        csr = csr_graph(_convert_from_eli)(eli, flags);
        eli_graph(_free)(eli);
    }

    return csr;
}

csr_graph_t *csr_graph(_copy)(const csr_graph_t *base, graph_flags_enum_t flags)
{
    assert(base != NULL);

    flags = copy_graph_flags(base->flags, flags);
    csr_graph_t *copy = csr_graph(_new)(flags);
    csr_graph(_set_size)(copy, base->vsize, base->esize);

    if (!(E_GRAPH_FLAG_DEG_I & flags))
        copy->deg_i = base->deg_i;
    if (!(E_GRAPH_FLAG_DEG_O & flags))
        copy->deg_o = base->deg_o;

    if (!(E_GRAPH_FLAG_TAG_V & flags))
        copy->vtag = base->vtag;
    if (!(E_GRAPH_FLAG_TAG_E & flags))
        copy->etag = base->etag;

    #ifdef GRAPH_V_TYPE
        if (!(E_GRAPH_FLAG_VAL_V & flags))
            copy->vval = base->vval;
    #endif

    #ifdef GRAPH_E_TYPE
        if (!(E_GRAPH_FLAG_VAL_E & flags))
            copy->eval = base->eval;
    #endif

    if (base->vcount > 0)
    {
        const graph_size_t vc = base->vcount;
        copy->vcount = vc;

        memcpy(copy->row_idx, base->row_idx, sizeof(*base->row_idx) * (vc + 1));

        if (E_GRAPH_FLAG_DEG_I & flags)
            memcpy(copy->deg_i, base->deg_i, sizeof(*base->deg_i) * vc);
        if (E_GRAPH_FLAG_DEG_O & flags)
            memcpy(copy->deg_o, base->deg_o, sizeof(*base->deg_o) * vc);
        if (E_GRAPH_FLAG_TAG_V & flags)
            memcpy(copy->vtag, base->vtag, sizeof(*base->vtag) * vc);

        #ifdef GRAPH_V_TYPE
            if (E_GRAPH_FLAG_VAL_V & flags)
                memcpy(copy->vval, base->vval, sizeof(*base->vval) * vc);
        #endif
    }

    if (base->ecount > 0)
    {
        const graph_size_t ec = base->ecount;
        copy->ecount = ec;

        memcpy(copy->col_idx, base->col_idx, sizeof(*base->col_idx) * ec);

        if (E_GRAPH_FLAG_TAG_E & flags)
            memcpy(copy->etag, base->etag, sizeof(*base->etag) * ec);

        #ifdef GRAPH_E_TYPE
            if (E_GRAPH_FLAG_VAL_E & flags)
                memcpy(copy->eval, base->eval, sizeof(*base->eval) * ec);
        #endif
    }

    return copy;
}

csr_graph_t *csr_graph(_mapped_copy_with_deg)(const csr_graph_t *base, const graph_size_t vcount, graph_size_t *local_deg_o, graph_flags_enum_t flags, const graph_map_vertex_func_t map_vertex, const graph_map_edge_func_t map_edge, void *map_arg)
{
    assert(base != NULL);
    assert(local_deg_o != NULL);
    if (map_vertex == NULL || map_edge == NULL)
        return csr_graph(_copy)(base, flags);

    flags = copy_graph_flags(base->flags, flags);
    csr_graph_t *copy = csr_graph(_new)(flags);

    csr_graph(_add_vertices)(copy, vcount);

    // Subset sum to get new row indices
    copy->row_idx[0] = 0;
    csr_forall_vertices(v, copy)
    {
        copy->row_idx[v + 1] = copy->row_idx[v] + local_deg_o[v];
    }

    copy->ecount = copy->row_idx[vcount];
    csr_graph(_set_size)(copy, copy->vsize, copy->ecount);

    graph_size_t src = 0;
    graph_size_t dst = 0;

    // Copy vertex values and tags
    if (E_GRAPH_FLAG_TAG_V & flags)
        csr_forall_vertices_par(v, base, private(src, dst))
        {
            if (map_vertex(v, &dst, map_arg))
                copy->vtag[dst] = base->vtag[v];
        }

    #ifdef GRAPH_V_TYPE
        if (E_GRAPH_FLAG_VAL_V & flags)
            csr_forall_vertices_par(v, base, private(src, dst))
            {
                if (map_vertex(v, &dst, map_arg))
                    copy->vval[dst] = base->vval[v];
            }
    #endif

    // Copy edges values
    csr_forall_edges(e,old_src,old_dst,base)
    {
        if (map_edge(old_src, old_dst, &src, &dst, map_arg))
        {
            assert(local_deg_o[src] > 0);

            // Invariant: row_idx[src+1] = row_idx[src] + local_deg_o[src]
            const graph_size_t eid = copy->row_idx[src + 1] - local_deg_o[src]--;

            copy->col_idx[eid] = dst;

            if (E_GRAPH_FLAG_TAG_E & flags)
                copy->etag[eid] = base->etag[e];

            #ifdef GRAPH_E_TYPE
                if (E_GRAPH_FLAG_VAL_E & flags)
                    copy->eval[eid] = base->eval[e];
            #endif
        }
    }

    if (E_GRAPH_FLAG_SORT & flags)
    {
        csr_forall_vertices_par(v, copy, /*no omp params*/)
        {
            const graph_size_t efr = copy->row_idx[v];
            const graph_size_t eto = copy->row_idx[v+1];
            if (efr >= eto)
                continue;
            qsort(&copy->col_idx[efr], eto-efr, sizeof(graph_size_t), _reorder_col_idx_compare);
        }
    }

    csr_graph(_recalc_managed_degrees)(copy, /*zero_mem = */false);
    return copy;
}

csr_graph_t *csr_graph(_mapped_copy)(const csr_graph_t *base, const graph_flags_enum_t flags, const graph_map_vertex_func_t map_vertex, const graph_map_edge_func_t map_edge, void *map_arg)
{
    assert(base != NULL);
    if (map_vertex == NULL || map_edge == NULL)
        return csr_graph(_copy)(base, flags);

    graph_size_t vcount = 0;
    graph_size_t src = 0;
    graph_size_t dst = 0;

    // Determine new vcount
    csr_forall_vertices_par(v, base, reduction(max:vcount) private(src, dst))
    {
        if (map_vertex(v, &dst, map_arg) && dst >= vcount)
            vcount = dst + 1;
    }

    // New out degree count
    graph_size_t *tmp = (graph_size_t*) memory_calloc(sizeof(graph_size_t) * vcount);

    // Count new number of edges to determine new ecount
    csr_forall_edges(old_src, old_dst, base)
    {
        if (map_edge(old_src, old_dst, &src, &dst, map_arg))
        {
            assert(src < vcount);
            tmp[src]++;
        }
    }

    csr_graph_t *copy = csr_graph(_mapped_copy_with_deg)(base, vcount, tmp, flags, map_vertex, map_edge, map_arg);

    memory_free(tmp);
    return copy;
}

csr_graph_t *csr_graph(_sorted_copy)(const csr_graph_t *base, const graph_flags_enum_t flags)
{
    assert(base != NULL);
    if (base->vcount < 1 || base->ecount < 1)
        return csr_graph(_copy)(base, flags);

    graph_size_t *idx = memory_talloc(graph_size_t, base->vcount);
    graph_size_t *deg = memory_talloc(graph_size_t, base->vcount);

    csr_forall_vertices_par(v, base, /*no omp params*/)
    {
        idx[v] = v;
        deg[v] = base->row_idx[v+1] - base->row_idx[v];
    }

    qsort_r(idx, base->vcount, sizeof(*idx), _reorder_by_degree_compare, deg);
    
    SWAP_VALUES(idx, deg)
    csr_forall_vertices_par(new_idx, base, /*no omp params*/)
    {
        const graph_size_t old_idx = deg[new_idx];
        idx[old_idx] = new_idx;
        deg[new_idx] = base->row_idx[old_idx+1] - base->row_idx[old_idx];
    }

    csr_forall_vertices_par(v, base, /*no omp params*/)
    {
        deg[idx[v]] = base->row_idx[v+1] - base->row_idx[v];
    }

    csr_graph_t *copy = csr_graph(_mapped_copy_with_deg)(
        base,
        base->vcount,
        deg,
        flags,
        (graph_map_vertex_func_t)_idx_mapping_fun_v,
        (graph_map_edge_func_t)_idx_mapping_fun_e,
        idx
    );

    memory_free(deg);
    memory_free(idx);
    return copy;
}

eli_graph_t *csr_graph(_get_eli_representation)(csr_graph_t *graph)
{
    assert(graph != NULL);

    graph_size_t *edge_from = memory_pinned_talloc_if(E_GRAPH_FLAG_PIN & graph->flags, *graph->col_idx, graph->esize);

    csr_forall_edges_par(e, src, dst, graph, /*no omp params*/)
    {
        assert(src < graph->vcount);
        edge_from[e] = src;
    }

    eli_graph_t *eli = eli_graph(_new_ex)(edge_from, graph->col_idx, graph->ecount, (graph_flags_enum_t) (graph->flags & ELI_VALID_FLAGS_MASK));

    eli->esize = graph->esize;
    eli->etag  = graph->etag;
    #ifdef GRAPH_E_TYPE
        eli->eval = graph->eval;
    #endif

    return eli;
}

static void csr_graph(_set_ridx_from_eli)(csr_graph_t *graph, const eli_graph_t *eli)
{
    assert(graph != NULL);
    assert(eli != NULL);

    memset(graph->row_idx, 0, sizeof(*graph->row_idx) * (graph->vcount + 1));
    eli_forall_edges(src, dst, eli)
    {
        graph->row_idx[src + 1]++; // Count degrees
    }

    // Subset sum on out degrees
    csr_forall_vertices(v, graph)
    {
        graph->row_idx[v + 1] += graph->row_idx[v];
    }

    graph->ecount = eli->ecount;
}

csr_graph_t *csr_graph(_convert_from_eli)(eli_graph_t *base, graph_flags_enum_t flags)
{
    assert(base != NULL);

    csr_graph_t *graph = csr_graph(_new)((graph_flags_enum_t) (copy_graph_flags(base->flags, flags) | (~ELI_VALID_FLAGS_MASK & flags)));
    if (base->ecount < 1)
        return graph;

    if (!(E_GRAPH_FLAG_SORT & base->flags))
        eli_graph(_sort)(base);

    graph_size_t vmax = 0;
    eli_graph(_get_vertex_range)(base, NULL, &vmax);

    csr_graph(_add_vertices)(graph, vmax + 1);
    csr_graph(_set_ridx_from_eli)(graph, base);

    graph->esize  = base->esize;
    graph->ecount = base->ecount;
    base->ecount = 0;

    graph->col_idx = base->eto;
    base->eto = NULL;

    graph->etag = base->etag;
    base->etag = NULL;

    #ifdef GRAPH_E_TYPE
        graph->eval = base->eval;
        base->eval = NULL;
    #endif

    eli_graph(_set_size)(base, 0);
    csr_graph(_recalc_managed_degrees)(graph, /*zero_mem = */false);

    return graph;
}

csr_graph_t *csr_graph(_copy_from_eli)(eli_graph_t *base, graph_flags_enum_t flags)
{
    assert(base != NULL);

    csr_graph_t *graph = csr_graph(_new)((graph_flags_enum_t) (copy_graph_flags(base->flags, flags) | (~ELI_VALID_FLAGS_MASK & flags)));
    if (base->ecount < 1)
        return graph;

    if (!(E_GRAPH_FLAG_SORT & base->flags))
        eli_graph(_sort)(base);

    graph_size_t vmax = 0;
    eli_graph(_get_vertex_range)(base, NULL, &vmax);

    csr_graph(_grow)(graph, vmax + 1, base->ecount);
    csr_graph(_add_vertices)(graph, vmax + 1);
    csr_graph(_set_ridx_from_eli)(graph, base);
    graph->ecount = base->ecount;

    memcpy(graph->col_idx, base->eto, sizeof(*graph->col_idx) * base->ecount);
    if (E_GRAPH_FLAG_TAG_E & flags)
        memcpy(graph->etag, base->etag, sizeof(*graph->etag) * base->ecount);

    #ifdef GRAPH_E_TYPE
        if (E_GRAPH_FLAG_VAL_E & flags)
            memcpy(graph->eval, base->eval, sizeof(*graph->eval) * base->ecount);
    #endif

    csr_graph(_recalc_managed_degrees)(graph, /*zero_mem = */false);
    return graph;
}

void csr_graph(_free_eli_representation)(csr_graph_t *graph, eli_graph_t *eli)
{
    assert(graph != NULL);
    assert(eli != NULL);

    if (graph->col_idx != eli->eto || graph->ecount != eli->ecount)
    {
        graph->esize   = eli->esize;
        graph->col_idx = eli->eto;
        graph->etag    = eli->etag;
        #ifdef GRAPH_E_TYPE
            graph->eval = eli->eval;
        #endif

        csr_graph(_set_ridx_from_eli)(graph, eli);
        csr_graph(_recalc_managed_degrees)(graph, /*zero_mem =*/true);
    }

    eli->eto = NULL;
    eli->etag = NULL;
    #ifdef GRAPH_E_TYPE
        eli->eval = NULL;
    #endif

    eli_graph(_free)(eli);
}

void csr_graph(_free)(csr_graph_t *graph)
{
    assert(graph != NULL);
    const bool pinned = E_GRAPH_FLAG_PIN & graph->flags;

    memory_pinned_free_if(pinned, graph->row_idx);
    memory_pinned_free_if(pinned, graph->col_idx);

    if (E_GRAPH_FLAG_DEG_I & graph->flags)
        memory_pinned_free_if(pinned, graph->deg_i);
    if (E_GRAPH_FLAG_DEG_O & graph->flags)
        memory_pinned_free_if(pinned, graph->deg_o);

    if (E_GRAPH_FLAG_TAG_V & graph->flags)
        memory_pinned_free_if(pinned, graph->vtag);
    if (E_GRAPH_FLAG_TAG_E & graph->flags)
        memory_pinned_free_if(pinned, graph->etag);

    #ifdef GRAPH_V_TYPE
        if (E_GRAPH_FLAG_VAL_V & graph->flags)
            memory_pinned_free_if(pinned, graph->vval);
    #endif

    #ifdef GRAPH_E_TYPE
        if (E_GRAPH_FLAG_VAL_E & graph->flags)
            memory_pinned_free_if(pinned, graph->eval);
    #endif

    memory_free((void*)graph);
}

void csr_graph(_clear)(csr_graph_t *graph)
{
    assert(graph != NULL);

    graph->vcount = 0;
    graph->ecount = 0;
    csr_graph(_set_size)(graph, 0, 0);
}

void csr_graph(_clear_tags)(csr_graph_t *graph)
{
    assert(graph != NULL);

    if (graph->vtag != NULL)
        memset(graph->vtag, 0, sizeof(*graph->vtag) * graph->vcount);
    if (graph->etag != NULL)
        memset(graph->etag, 0, sizeof(*graph->etag) * graph->ecount);
}

#if defined(GRAPH_V_TYPE) || defined(GRAPH_E_TYPE)
    void csr_graph(_clear_values)(csr_graph_t *graph)
    {
        assert(graph != NULL);

        #ifdef GRAPH_V_TYPE
            if (graph->vval != NULL)
                memset(graph->vval, 0, sizeof(*graph->vval) * graph->vcount);
        #endif

        #ifdef GRAPH_E_TYPE
            if (graph->eval != NULL)
                memset(graph->eval, 0, sizeof(*graph->eval) * graph->ecount);
        #endif
    }
#endif

bool csr_graph(_equals)(const csr_graph_t *restrict first, const csr_graph_t *restrict second)
{
    if (first == second)
        return true;

    assert(first != NULL && second != NULL);

    const graph_size_t vcount = first->vcount;
    const graph_size_t ecount = first->ecount;

    return (vcount == second->vcount)
        && (ecount == second->ecount)
        && (first->row_idx == second->row_idx || memcmp(first->row_idx, second->row_idx, sizeof(*first->row_idx) * (vcount + 1)))
        && (first->col_idx == second->col_idx || memcmp(first->col_idx, second->col_idx, sizeof(*first->col_idx) * ecount));
}

void csr_graph(_write)(const csr_graph_t * graph, FILE *stream)
{
    assert(graph != NULL);
    assert(stream != NULL);

    fprintf(stream, "%zu %zu %1u%1u%1u\n",
        (size_t)graph->vcount,
        (size_t)graph->ecount / 2,
        false,
        E_GRAPH_FLAG_TAG_V & graph->flags,
        E_GRAPH_FLAG_TAG_E & graph->flags
    );

    csr_forall_vertices(src, graph)
    {
        if (E_GRAPH_FLAG_TAG_V & graph->flags)
            fprintf(stream, "%zu ", (size_t)graph->vtag[src]);

        if (E_GRAPH_FLAG_TAG_E & graph->flags)
            csr_forall_out_edges(e, dst, src, graph)
            {
                fprintf(stream, "%zu %zu ", (size_t)dst + 1, (size_t)graph->etag[e]);
            }
        else
            csr_forall_out_edges(dst, src, graph)
            {
                fprintf(stream, "%zu ", (size_t)dst + 1);
            }

        fputc('\n', stream);
    }
}

bool csr_graph(_write_file)(csr_graph_t * graph, const char *const filename, const char *const force_ext)
{
    assert(graph != NULL);

    const char *ext = (force_ext != NULL) ? force_ext : file_extension(filename);
    if (!ext)
        return false;

    if (strcasecmp(ext, "dimacs") == 0 || strcasecmp(ext, "csr") == 0 || strcasecmp(ext, "csc") == 0)
    {
        FILE* file = (filename == NULL || strcmp(filename, "-") == 0)
            ? stdout
            : fopen(filename, "w+");

        if (file)
        {
            if (strcasecmp(ext, "dimacs") == 0)
            {
                csr_graph(_remove_self_loops)(graph);
                csr_graph(_to_directed)(graph);
                csr_graph(_remove_dup_edges)(graph, graph_merge_tag_add, NULL);
                csr_graph(_remove_unconnected)(graph);
            }
            else if (strcasecmp(ext, "csc") == 0)
            {
                csr_graph(_transpose)(graph);
            }

            csr_graph(_write)(graph, file);
            if (file != stdout)
                fclose(file);

            if (strcasecmp(ext, "csc") == 0)
            {
                csr_graph(_transpose)(graph);
            }

            return true;
        }
    }
    else
    {
        eli_graph_t *eli = csr_graph(_get_eli_representation)(graph);
        const bool res = eli_graph(_write_file)(eli, filename, force_ext);
        csr_graph(_free_eli_representation)(graph, eli);
        return res;
    }

    return false;
}

size_t csr_graph(_byte_size)(csr_graph_t *graph, bool allocated)
{
    assert(graph != NULL);

    const graph_size_t v = (allocated) ? graph->vsize : graph->vcount;
    const graph_size_t e = (allocated) ? graph->esize : graph->ecount;

    size_t s = sizeof(*graph)
        + (sizeof(*graph->row_idx) * (v + 1))
        + (sizeof(*graph->col_idx) * e);

    if (E_GRAPH_FLAG_DEG_I & graph->flags)
        s += sizeof(*graph->deg_i) * v;
    if (E_GRAPH_FLAG_DEG_O & graph->flags)
        s += sizeof(*graph->deg_o) * v;

    if (E_GRAPH_FLAG_TAG_V & graph->flags)
        s += sizeof(*graph->vtag) * v;
    if (E_GRAPH_FLAG_TAG_E & graph->flags)
        s += sizeof(*graph->etag) * e;

    #ifdef GRAPH_V_TYPE
        if (E_GRAPH_FLAG_VAL_V & graph->flags)
            s += sizeof(*graph->vval) * v;
    #endif

    #ifdef GRAPH_E_TYPE
        if (E_GRAPH_FLAG_VAL_E & graph->flags)
            s += sizeof(*graph->eval) * e;
    #endif

    return s;
}

graph_size_t csr_graph(_align_edges)(csr_graph_t *graph, const graph_size_t alignment, graph_size_t dst)
{
    assert(graph != NULL);

    if (alignment <= 1 || graph->ecount < 1)
        return 0;

    if (dst >= graph->vcount)
        dst = graph->vcount - 1;

    graph_size_t grow = 0;
    csr_forall_vertices(src, graph)
    {
        const graph_size_t count = graph->row_idx[src + 1] - graph->row_idx[src];
        const graph_size_t shift = ROUND_TO_MULT(count, alignment) - count;
        grow += shift;
    }

    csr_graph(_grow)(graph, 0, grow);

    csr_forall_vertices(src, graph)
    {
        const graph_size_t count = graph->row_idx[src + 1] - graph->row_idx[src];
        const graph_size_t shift = ROUND_TO_MULT(count, alignment) - count;

        if (shift > 0)
            csr_graph(_add_edges)(graph, src, dst, shift);
    }

    return grow;
}

void csr_graph(_set_size)(csr_graph_t *graph, graph_size_t vsize, graph_size_t esize)
{
    assert(graph != NULL);

    const bool         pinned    = E_GRAPH_FLAG_PIN & graph->flags;
    const graph_size_t alignment = memory_get_default_alignment();

    vsize = ROUND_TO_MULT(MAX(vsize, graph->vcount), alignment);
    esize = ROUND_TO_MULT(MAX(esize, graph->ecount), alignment);

    if (graph->vsize != vsize)
    {
        graph->row_idx = memory_pinned_retalloc_if(pinned, graph->row_idx, graph->vsize+1, vsize+1);

        graph->deg_i = (E_GRAPH_FLAG_DEG_I & graph->flags)
            ? memory_pinned_retalloc_if(pinned, graph->deg_i, graph->vsize, vsize)
            : NULL;
        graph->deg_o = (E_GRAPH_FLAG_DEG_O & graph->flags)
            ? memory_pinned_retalloc_if(pinned, graph->deg_o, graph->vsize, vsize)
            : NULL;
        graph->vtag = (E_GRAPH_FLAG_TAG_V & graph->flags)
            ? memory_pinned_retalloc_if(pinned, graph->vtag, graph->vsize, vsize)
            : NULL;

        #ifdef GRAPH_V_TYPE
            graph->vval = (E_GRAPH_FLAG_VAL_V & graph->flags)
                ? memory_pinned_retalloc_if(pinned, graph->vval, graph->vsize, vsize)
                : NULL;
        #endif

        graph->vsize = vsize;
    }

    if (graph->esize != esize)
    {
        graph->col_idx = memory_pinned_retalloc_if(pinned, graph->col_idx, graph->esize, esize);

        graph->etag = (E_GRAPH_FLAG_TAG_E & graph->flags)
            ? memory_pinned_retalloc_if(pinned, graph->etag, graph->esize, esize)
            : NULL;

        #ifdef GRAPH_E_TYPE
            graph->eval = (E_GRAPH_FLAG_VAL_E & graph->flags)
                ? memory_pinned_retalloc_if(pinned, graph->eval, graph->esize, esize)
                : NULL;
        #endif

        graph->esize = esize;
    }
}

void csr_graph(_grow)(csr_graph_t *graph, graph_size_t vgrow, graph_size_t egrow)
{
    assert(graph != NULL);

    vgrow = (graph->vcount + vgrow <= graph->vsize)
        ? graph->vsize
        : MAX((graph_size_t)(graph->vsize * GRAPH_V_GROW), graph->vcount + vgrow);

    egrow = (graph->ecount + egrow <= graph->esize)
         ? graph->esize
         : MAX((graph_size_t)(graph->esize * GRAPH_E_GROW), graph->ecount + egrow);

    csr_graph(_set_size)(graph, vgrow, egrow);
}

void csr_graph(_shrink)(csr_graph_t *graph)
{
    csr_graph(_set_size)(graph, graph->vcount, graph->ecount);
}

bool csr_graph(_toggle_flag)(csr_graph_t *graph, const graph_flags_enum_t flag, const bool enable)
{
    assert(graph != NULL);

    if ((graph->flags & flag) != enable)
    {
        switch(flag)
        {
            case E_GRAPH_FLAG_PIN:
            {
                const memory_manager_enum_t mm_fr = (enable) ? E_MM_DEFAULT : memory_get_default_pinned_manager();
                const memory_manager_enum_t mm_to = (enable) ? memory_get_default_pinned_manager() : E_MM_DEFAULT;

                #define _toggle_pinned_(ptr,size) do {if (ptr) {ptr = CAST_TO_TYPE_OF(ptr) memory_pinned_realloc_managers(mm_fr, mm_to, ptr, (size) * sizeof(*ptr)); }} while(false)

                _toggle_pinned_(graph->row_idx, graph->vsize + 1);
                _toggle_pinned_(graph->col_idx, graph->esize);

                _toggle_pinned_(graph->deg_i, graph->vsize);
                _toggle_pinned_(graph->deg_o, graph->vsize);

                _toggle_pinned_(graph->vtag, graph->vsize);
                _toggle_pinned_(graph->etag, graph->esize);

                #ifdef GRAPH_V_TYPE
                    _toggle_pinned_(graph->vval, graph->vsize);
                #endif

                #ifdef GRAPH_E_TYPE
                    _toggle_pinned_(graph->eval, graph->esize);
                #endif

                #undef _toggle_pinned_
                break;
            }

            case E_GRAPH_FLAG_DEG_I:
            {
                memory_pinned_toggle_if(graph->flags & E_GRAPH_FLAG_PIN, enable, graph->deg_i, graph->vsize);
                if (enable)
                {
                    memset(graph->deg_i, 0, sizeof(*graph->deg_i) * graph->vcount);
                    csr_graph(_calc_vertex_degrees_in)(graph, graph->deg_i);
                }
                break;
            }

            case E_GRAPH_FLAG_DEG_O:
            {
                memory_pinned_toggle_if(graph->flags & E_GRAPH_FLAG_PIN, enable, graph->deg_o, graph->vsize);
                if (enable)
                {
                    memset(graph->deg_o, 0, sizeof(*graph->deg_o) * graph->vcount);
                    csr_graph(_calc_vertex_degrees_out)(graph, graph->deg_o);
                }
                break;
            }

            case E_GRAPH_FLAG_TAG_V: memory_pinned_toggle_if(graph->flags & E_GRAPH_FLAG_PIN, enable, graph->vtag, graph->vsize); break;
            case E_GRAPH_FLAG_TAG_E: memory_pinned_toggle_if(graph->flags & E_GRAPH_FLAG_PIN, enable, graph->etag, graph->esize); break;

            #ifdef GRAPH_V_TYPE
                case E_GRAPH_FLAG_VAL_V: memory_pinned_toggle_if(graph->flags & E_GRAPH_FLAG_PIN, enable, graph->vval, graph->vsize); break;
            #endif

            #ifdef GRAPH_E_TYPE
                case E_GRAPH_FLAG_VAL_E: memory_pinned_toggle_if(graph->flags & E_GRAPH_FLAG_PIN, enable, graph->eval, graph->esize); break;
            #endif

            default:
                return false;
        }

        if (enable)
            graph->flags = (graph_flags_enum_t) (graph->flags | flag);
        else
            graph->flags = (graph_flags_enum_t) (graph->flags & ~flag);
    }

    return true;
}

void csr_graph(_transpose)(csr_graph_t *graph)
{
    assert(graph != NULL);
    if (graph->vcount < 1 || graph->ecount < 1)
        return;

    eli_graph_t *eli = csr_graph(_get_eli_representation)(graph);
    eli_graph(_transpose)(eli);
    csr_graph(_free_eli_representation)(graph, eli);
}

void csr_graph(_to_directed)(csr_graph_t *graph)
{
    assert(graph != NULL);
    if (graph->vcount < 1 || graph->ecount < 1)
        return;

    eli_graph_t *eli = csr_graph(_get_eli_representation)(graph);
    eli_graph(_to_directed)(eli);
    csr_graph(_free_eli_representation)(graph, eli);
}

graph_size_t csr_graph(_to_undirected)(csr_graph_t *graph)
{
    assert(graph != NULL);
    graph_size_t removed = 0;

    for (graph_size_t v = 0, efr = 0; v < graph->vcount; v++)
    {
        const graph_size_t eto = graph->row_idx[v+1];

        for (graph_size_t e = efr; e < eto; e++)
        {
            if (graph->col_idx[e] > v)
            {
                if (graph->deg_i != NULL) graph->deg_i[graph->col_idx[e]]--;
                if (graph->deg_o != NULL) graph->deg_o[v]--;
                removed++;
            }
            else if (removed > 0)
            {
                graph->col_idx[e - removed] = graph->col_idx[e];
                if (graph->etag != NULL) graph->etag[e - removed] = graph->etag[e];
                #ifdef GRAPH_E_TYPE
                    if (graph->eval != NULL) graph->eval[e - removed] = graph->eval[e];
                #endif
            }
        }

        if (removed > 0)
            graph->row_idx[v+1] -= removed;

        efr = eto;
    }

    graph->ecount -= removed;
    return removed;
}

graph_size_t csr_graph(_remove_dup_edges)(csr_graph_t *graph, const graph_merge_tag_func_t merge_tag, void *merge_arg)
{
    assert(graph != NULL);
    graph_size_t removed = 0;

    for (graph_size_t v = 0, efr = 0; v < graph->vcount; v++)
    {
        const graph_size_t eto = graph->row_idx[v+1];

        for (graph_size_t e = efr; e < eto; e++)
        {
            if (e > efr && graph->col_idx[e-1] == graph->col_idx[e])
            {
                if (graph->deg_i != NULL) graph->deg_i[graph->col_idx[e]]--;
                if (graph->deg_o != NULL) graph->deg_o[v]--;
                if (merge_tag != NULL && graph->etag != NULL)
                    graph->etag[e-1] = merge_tag(graph->etag[e-1], graph->etag[e], merge_arg);
                removed++;
            }
            else if (removed > 0)
            {
                graph->col_idx[e - removed] = graph->col_idx[e];
                if (graph->etag != NULL) graph->etag[e - removed] = graph->etag[e];
                #ifdef GRAPH_E_TYPE
                    if (graph->eval != NULL) graph->eval[e - removed] = graph->eval[e];
                #endif
            }
        }

        if (removed > 0)
            graph->row_idx[v+1] -= removed;

        efr = eto;
    }

    graph->ecount -= removed;
    return removed;
}

graph_size_t csr_graph(_remove_self_loops)(csr_graph_t *graph)
{
    assert(graph != NULL);
    graph_size_t removed = 0;

    for (graph_size_t v = 0, efr = 0; v < graph->vcount; v++)
    {
        const graph_size_t eto = graph->row_idx[v+1];

        for (graph_size_t e = efr; e < eto; e++)
            if (graph->col_idx[e] == v)
            {
                if (graph->deg_i != NULL) graph->deg_i[v]--;
                if (graph->deg_o != NULL) graph->deg_o[v]--;
                removed++;
            }
            else if (removed > 0)
            {
                graph->col_idx[e - removed] = graph->col_idx[e];
                if (graph->etag != NULL) graph->etag[e - removed] = graph->etag[e];
                #ifdef GRAPH_E_TYPE
                    if (graph->eval != NULL) graph->eval[e - removed] = graph->eval[e];
                #endif
            }

        if (removed > 0)
            graph->row_idx[v+1] -= removed;

        efr = eto;
    }

    graph->ecount -= removed;
    return removed;
}

graph_size_t csr_graph(_remove_unconnected)(csr_graph_t *graph)
{
    assert(graph != NULL);

    graph_size_t *deg_i = graph->deg_i;
    graph_size_t *shift = memory_talloc(graph_size_t, graph->vcount);

    if (deg_i == NULL && graph->vcount > 0)
    {
        deg_i = shift;
        memset(deg_i, 0, graph->vcount * sizeof(*deg_i));

        csr_forall_edges_par(e, graph, /*no omp params*/)
        {
            if (deg_i[graph->col_idx[e]] == 0)
                deg_i[graph->col_idx[e]] = 1;
        }
    }

    graph_size_t removed = 0;
    csr_forall_vertices(v, graph)
    {
        if (deg_i[v] == 0 && graph->row_idx[v] == graph->row_idx[v+1])
            removed++;
        else
        {
            shift[v] = v - removed;
            if (removed > 0)
            {
                graph->row_idx[v - removed] = graph->row_idx[v];
                if (graph->vtag != NULL) graph->vtag[v - removed] = graph->vtag[v];
                #ifdef GRAPH_V_TYPE
                    if (graph->vval != NULL) graph->vval[v - removed] = graph->vval[v];
                #endif
            }
        }
    }

    if (removed > 0)
    {
        csr_forall_edges(e, graph)
        {
            graph->col_idx[e] = shift[graph->col_idx[e]];
        }

        graph->row_idx[graph->vcount - removed] = graph->row_idx[graph->vcount];
        graph->vcount -= removed;
    }

    memory_free(shift);
    return removed;
}

bool csr_graph(_get_edge_index)(const csr_graph_t *graph, const graph_size_t src, const graph_size_t dst, graph_size_t *res)
{
    assert(graph != NULL);
    assert(src < graph->vcount);

    const graph_size_t lo = graph->row_idx[src];
    const graph_size_t hi = graph->row_idx[src + 1];

    if (hi > lo)
        return BINARY_SEARCH(dst, graph->col_idx, lo, hi - 1, *res);
    else if (res != NULL)
        *res = hi;

    return false;
}

void csr_graph(_insert_vertices)(csr_graph_t *graph, graph_size_t idx, const graph_size_t count)
{
    assert(graph != NULL);

    if (count > 0)
    {
        const graph_size_t len = graph->vcount;
        if (idx > len)
            idx = len;

        csr_graph(_grow)(graph, count, 0);
        graph->vcount += count;

        // Move old indices to create space
        memmove(&graph->row_idx[idx+count], &graph->row_idx[idx], sizeof(*graph->row_idx) * (len + 1 - idx));

        // Initialize new vertex row indices
        const graph_size_t row_idx = graph->row_idx[idx];

        OMP_PRAGMA(omp parallel for)
        for (graph_size_t v = 0; v < count; v++)
            graph->row_idx[idx + v] = row_idx;

        // Update edge destinations
        if (idx < len)
            csr_forall_edges_par(e, graph, /*no omp params*/)
            {
                if (graph->col_idx[e] >= idx)
                    graph->col_idx[e] += count;
            }

        // Initialize new vertex degrees or invalidate if not self-managed
        if (graph->deg_i != NULL)
        {
            memmove(&graph->deg_i[idx+count], &graph->deg_i[idx], sizeof(*graph->deg_i) * (len - idx));
            memset(&graph->deg_i[idx], 0, sizeof(*graph->deg_i) * count);
        }

        if (graph->deg_o != NULL)
        {
            memmove(&graph->deg_o[idx+count], &graph->deg_o[idx], sizeof(*graph->deg_o) * (len - idx));
            memset(&graph->deg_o[idx], 0, sizeof(*graph->deg_o) * count);
        }

        if (graph->vtag != NULL)
        {
            memmove(&graph->vtag[idx+count], &graph->vtag[idx], sizeof(*graph->vtag) * (len - idx));
            memset(&graph->vtag[idx], 0, sizeof(*graph->vtag) * count);
        }

        #ifdef GRAPH_V_TYPE
            // Initialize new vertex values or invalidate if not self-managed
            if (graph->vval != NULL)
            {
                memmove(&graph->vval[idx+count], &graph->vval[idx], sizeof(*graph->vval) * (len - idx));
                memset(&graph->vval[idx], 0, sizeof(*graph->vval) * count);
            }
        #endif
    }
}

void csr_graph(_insert_vertex)(csr_graph_t *graph, graph_size_t idx)
{
    csr_graph(_insert_vertices)(graph, idx, 1);
}

graph_size_t csr_graph(_add_vertices)(csr_graph_t *graph, const graph_size_t count)
{
    assert(graph != NULL);

    const graph_size_t idx = graph->vcount;
    csr_graph(_insert_vertices)(graph, idx, count);

    return idx;
}

graph_size_t csr_graph(_add_vertex)(csr_graph_t *graph)
{
    return csr_graph(_add_vertices)(graph, 1);
}

graph_size_t csr_graph(_add_edges)(csr_graph_t *graph, const graph_size_t src, const graph_size_t dst, const graph_size_t count)
{
    graph_size_t idx;
    csr_graph(_get_edge_index)(graph, src, dst, &idx);

    if (count > 0)
    {
        const graph_size_t len = graph->ecount;
        if (idx > len)
            idx = len;

        csr_graph(_grow)(graph, 0, count);
        graph->ecount += count;

        // Move old indices to create space
        memmove(&graph->col_idx[idx+count], &graph->col_idx[idx], sizeof(*graph->col_idx) * (len - idx));

        // Initialize new edges' destinations
        for (graph_size_t e = 0; e < count; e++)
            graph->col_idx[idx + e] = dst;

        if (graph->etag != NULL)
        {
            memmove(&graph->etag[idx+count], &graph->etag[idx], sizeof(*graph->etag) * (len - idx));
            memset(&graph->etag[idx], 0, sizeof(*graph->etag) * count);
        }

        // Initialize new edge values or invalidate if not self-managed
        #ifdef GRAPH_E_TYPE
            if (graph->eval != NULL)
            {
                memmove(&graph->eval[idx+count], &graph->eval[idx], sizeof(*graph->eval) * (len - idx));
                memset(&graph->eval[idx], 0, sizeof(*graph->eval) * count);
            }
        #endif

        // Update row indices
        OMP_PRAGMA(omp parallel for)
        for (graph_size_t v = src + 1; v <= graph->vcount; v++)
            graph->row_idx[v] += count;

        // Update vertex degree
        if (graph->deg_i != NULL)
            graph->deg_i[dst] += count;
        if (graph->deg_o != NULL)
            graph->deg_o[src] += count;
    }

    return idx;
}

graph_size_t csr_graph(_add_edge)(csr_graph_t *graph, const graph_size_t src, const graph_size_t dst)
{
    return csr_graph(_add_edges)(graph, src, dst, 1);
}

void csr_graph(_add_edgelist)(csr_graph_t *graph, const graph_size_t *src, const graph_size_t *dst, const graph_size_t count)
{
    csr_graph(_grow)(graph, 0, count);

    for (graph_size_t e = 0; e < count; e++)
        csr_graph(_add_edge)(graph, src[e], dst[e]);
}

void csr_graph(_calc_vertex_degrees)(const csr_graph_t *graph, graph_size_t *deg)
{
    csr_graph(_calc_vertex_degrees_in)(graph, deg);
    csr_graph(_calc_vertex_degrees_out)(graph, deg);
}

void csr_graph(_calc_vertex_degrees_in)(const csr_graph_t *graph, graph_size_t *restrict deg)
{
    assert(graph != NULL);
    assert(deg != NULL || graph->ecount == 0);

    csr_forall_edges(e, graph)
    {
        deg[graph->col_idx[e]]++;
    }
}

void csr_graph(_calc_vertex_degrees_out)(const csr_graph_t *graph, graph_size_t *restrict deg)
{
    assert(graph != NULL);
    assert(deg != NULL || graph->ecount == 0);

    csr_forall_vertices_par(v, graph, /*no omp params*/)
    {
        deg[v] += graph->row_idx[v + 1] - graph->row_idx[v];
    }
}

void csr_graph(_recalc_managed_degrees)(const csr_graph_t *graph, const bool zero_mem)
{
    if (graph->deg_i != NULL)
    {
        if (zero_mem)
            memset(graph->deg_i, 0, sizeof(*graph->deg_i) * graph->vcount);
        csr_graph(_calc_vertex_degrees_in)(graph, graph->deg_i);
    }
    if (graph->deg_o != NULL)
    {
        if (zero_mem)
            memset(graph->deg_o, 0, sizeof(*graph->deg_o) * graph->vcount);
        csr_graph(_calc_vertex_degrees_out)(graph, graph->deg_o);
    }
}

graph_size_t *csr_graph(_get_vertex_degrees)(csr_graph_t *graph)
{
    assert(graph != NULL);
    if (graph->vcount < 1)
        return NULL;

          graph_size_t *restrict res   = memory_talloc(graph_size_t, graph->vcount);
    const graph_size_t *restrict deg_i = csr_graph(_get_vertex_degrees_in)(graph);
    const graph_size_t *restrict deg_o = csr_graph(_get_vertex_degrees_out)(graph);

    csr_forall_vertices_par(v, graph, /*no omp params*/)
    {
        res[v] = deg_i[v] + deg_o[v];
    }

    return res;
}

const graph_size_t *csr_graph(_get_vertex_degrees_in)(csr_graph_t *graph)
{
    assert(graph != NULL);

    if (graph->deg_i == NULL && graph->vcount > 0)
        csr_graph(_toggle_flag)(graph, E_GRAPH_FLAG_DEG_I, true);

    return graph->deg_i;
}

const graph_size_t *csr_graph(_get_vertex_degrees_out)(csr_graph_t *graph)
{
    assert(graph != NULL);

    if (graph->deg_o == NULL && graph->vcount > 0)
        csr_graph(_toggle_flag)(graph, E_GRAPH_FLAG_DEG_O, true);

    return graph->deg_o;
}

graph_size_t csr_graph(_get_vertex_degree)(const csr_graph_t *graph, const graph_size_t idx)
{
    return csr_graph(_get_vertex_degree_in)(graph, idx)
         + csr_graph(_get_vertex_degree_out)(graph, idx);
}

graph_size_t csr_graph(_get_vertex_degree_in)(const csr_graph_t *graph, const graph_size_t idx)
{
    assert(graph != NULL);
    assert(idx < graph->vcount);

    if (graph->deg_i != NULL)
        return graph->deg_i[idx];

    graph_size_t count = 0;

    csr_forall_edges_par(e, graph, reduction(+:count))
    {
        if (graph->col_idx[e] == idx)
            count++;
    }

    return count;
}

graph_size_t csr_graph(_get_vertex_degree_out)(const csr_graph_t *graph, const graph_size_t idx)
{
    assert(graph != NULL);
    assert(idx < graph->vcount);

    return (graph->deg_o != NULL)
        ? graph->deg_o[idx]
        : graph->row_idx[idx + 1] - graph->row_idx[idx];
}

double csr_graph(_avg_clustering_coefficient)(csr_graph_t *graph)
{
    assert(graph != NULL);
    if (graph->vcount < 1)
        return 0.0;

    uint64_t triangles = 0;
    uint64_t triples   = 0;

    csr_forall_vertices_par(v1, graph, reduction(+:triples) reduction(+:triangles))
    {
        const graph_size_t deg = csr_graph(_get_vertex_degree_out)(graph, v1);
        triples += deg * (uint64_t)(deg - 1);

        csr_forall_out_edges(v2, v1, graph)
        {
            if (v1 == v2) continue;

            csr_forall_out_edges(v3, v2, graph)
            {
                if (v1 == v3 || v2 == v3) continue;

                graph_size_t idx;
                if (csr_graph(_get_edge_index)(graph, v1, v3, &idx))
                {
                    triangles++;
                }
            }
        }
    }

    return triangles / (double) triples;
}

double csr_graph(_avg_neighbor_degree_in)(csr_graph_t *graph)
{
    assert(graph != NULL);
    if (graph->vcount < 1)
        return 0.0;

    const graph_size_t *deg = csr_graph(_get_vertex_degrees_in)(graph);
          long double   sum = 0;

    csr_forall_vertices_par(efr, graph, reduction(+:sum))
    {
        csr_forall_out_edges(eto, efr, graph)
        {
            sum += deg[eto];
        }
    }

    return (double) (sum / graph->vcount);
}

double csr_graph(_avg_neighbor_degree_out)(csr_graph_t *graph)
{
    assert(graph != NULL);
    if (graph->vcount < 1)
        return 0.0;

    long double sum = 0;

    csr_forall_vertices_par(v, graph, reduction(+:sum))
    {
        const graph_size_t deg = csr_graph(_get_vertex_degree_out)(graph, v);
        sum += deg * deg;
    }

    return (double) (sum / graph->vcount);
}

double csr_graph(_degree_assortativity)(csr_graph_t *graph)
{
    assert(graph != NULL);
    if (graph->vcount < 1)
        return 0.0;

    const graph_size_t *deg_in = csr_graph(_get_vertex_degrees_in)(graph);

    long double sum_deg   = 0;
    long double sum_dego  = 0;
    long double sum_degi  = 0;
    long double ssum_dego = 0;
    long double ssum_degi = 0;

    csr_forall_vertices_par(efr, graph, reduction(+:sum_deg) reduction(+:sum_dego) reduction(+:sum_degi) reduction(+:ssum_dego) reduction(+:ssum_degi))
    {
        const graph_size_t dego = csr_graph(_get_vertex_degree_out)(graph, efr) - 1;

        csr_forall_out_edges(eto, efr, graph)
        {
            const graph_size_t degi = deg_in[eto] - 1;

            sum_deg   += dego * (long double)degi;
            sum_dego  += dego;
            sum_degi  += degi;
            ssum_dego += dego * (long double)dego;
            ssum_degi += degi * (long double)degi;
        }
    }

    printf("%Lf %Lf %Lf %Lf %Lf\n", sum_deg, sum_dego, sum_degi, ssum_dego, ssum_degi);

    const long double ec = graph->ecount;
    return (double) (((ec * sum_deg) - (sum_dego * sum_degi)) /
        sqrtl(((ec * ssum_dego) - (sum_dego * sum_dego)) * ((ec * ssum_degi) - (sum_degi * sum_degi)) + 1));
}