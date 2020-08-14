// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/eli.h"
#include "util/math.h"
#include "util/memory.h"
#include "vendor/graph500/graph_generator.h"
#include "vendor/graph500/utils.h"

#include <zlib.h>

#ifndef GRAPH_NAME
    #include "graph/template/eli.h"
#endif

static int _reorder_idx_compare(const void *a, const void *b);

eli_graph_t *eli_graph(_new_ex)(graph_size_t *restrict efr, graph_size_t *restrict eto, const graph_size_t ecount, const graph_flags_enum_t flags)
{
    assert(!(~ELI_VALID_FLAGS_MASK & flags));

    #ifndef GRAPH_E_TYPE
        assert(!(E_GRAPH_FLAG_VAL_E & flags));
    #endif

    eli_graph_t *graph = memory_talloc(eli_graph_t);
    assert(graph != NULL);

    graph->efr = efr;
    graph->eto = eto;
    graph->ecount = ecount;
    graph->esize  = ecount;

    graph->flags = flags;
    return graph;
}

eli_graph_t *eli_graph(_new)(const graph_flags_enum_t flags)
{
    return eli_graph(_new_ex)(NULL, NULL, 0, flags);
}

eli_graph_t *eli_graph(_new_random)(const graph_size_t vcount, graph_size_t ecount, const graph_flags_enum_t flags)
{
    eli_graph_t *graph = eli_graph(_new)(flags);

    const unsigned long long int max_vcount = (unsigned long long int)vcount * vcount;
    if (ecount > max_vcount)
        ecount = max_vcount;

    eli_graph(_grow)(graph, ecount);

    if (ecount < 1)
        return graph;

    const long double edge_p = ecount / (long double) max_vcount;
    const long double lp = logl(1.0 - edge_p);

    graph_size_t edge_frm = 0;
    graph_size_t edge_to  = 0;
    graph_size_t edge_idx = 0;

    while (true)
    {
        const long double lr = logl(1.0 - (rand() / (long double) RAND_MAX));
        edge_to += 1 + (int)(lr / lp);

        while ((edge_to >= vcount) && (edge_frm < vcount))
        {
            edge_frm++;
            edge_to -= vcount;
        }

        if (edge_frm >= vcount)
            break;
        else if (edge_idx < ecount)
        {
            graph->efr[edge_idx] = edge_frm;
            graph->eto[edge_idx] = edge_to;
            edge_idx++;
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

eli_graph_t *eli_graph(_new_regular)(const graph_size_t vcount, graph_size_t deg, graph_size_t stride, const graph_flags_enum_t flags)
{
    eli_graph_t *graph = eli_graph(_new)(flags);

    graph->ecount = vcount * deg;
    if (graph->ecount < 1)
        return graph;

    eli_graph(_grow)(graph, graph->ecount);

    OMP_PRAGMA(omp parallel for)
    for (graph_size_t v = 0; v < vcount; v++)
    {
        const graph_size_t start  = v * deg;

        for (graph_size_t d = 0; d < deg; d++)
        {
            graph->efr[start + d] = v;
            graph->eto[start + d] = ((v * stride) + (d * stride)) % vcount;
        }

        if ((E_GRAPH_FLAG_SORT & graph->flags) && ((v * stride) + ((deg - 1) * stride) >= vcount)) {
            qsort(&graph->eto[start], deg, sizeof(graph_size_t), _reorder_idx_compare);
        }
    }

    if (graph->etag != NULL)
        memset(graph->etag, 0, sizeof(*graph->etag) * graph->ecount);
    #ifdef GRAPH_E_TYPE
        if (graph->eval != NULL)
            memset(graph->eval, 0, sizeof(*graph->eval) * graph->ecount);
    #endif

    return graph;
}

eli_graph_t *eli_graph(_new_kronecker)(const graph_size_t vcount, graph_size_t ecount, const graph_flags_enum_t flags)
{
    eli_graph_t *graph = eli_graph(_new)((graph_flags_enum_t) (flags & ~E_GRAPH_FLAG_SORT));
    eli_graph(_grow)(graph, ecount);
    graph->ecount = ecount;

    packed_edge *buf = memory_talloc(packed_edge, ecount);
    assert(buf != NULL);

    static const uint64_t seed1 = 2, seed2 = 3;
    uint_fast32_t seed[5];
    make_mrg_seed(seed1, seed2, seed);
    generate_kronecker_range(seed, round(log2(vcount)), 0, ecount, buf);

    OMP_PRAGMA(omp parallel for)
    for (graph_size_t e = 0; e < ecount; e++)
    {
        graph->efr[e] = get_v0_from_edge(&buf[e]);
        graph->eto[e] = get_v1_from_edge(&buf[e]);
    }

    if (graph->etag != NULL)
        memset(graph->etag, 0, sizeof(*graph->etag) * graph->ecount);
    #ifdef GRAPH_E_TYPE
        if (graph->eval != NULL)
            memset(graph->eval, 0, sizeof(*graph->eval) * graph->ecount);
    #endif

    eli_graph(_toggle_flag)(graph, E_GRAPH_FLAG_SORT, E_GRAPH_FLAG_SORT & flags);
    return graph;
}

eli_graph_t *eli_graph(_deserialize)(FILE *stream)
{
    assert(stream != NULL);

    gzFile f = gzdopen(dup(fileno(stream)), "rb");
    assert(f != NULL);

    uint64_t flags;  gzread(f, &flags,  sizeof(flags));
    uint64_t ecount; gzread(f, &ecount, sizeof(ecount));

    eli_graph_t *graph = eli_graph(_new)((graph_flags_enum_t) (flags));
    eli_graph(_grow)(graph, (graph_size_t) ecount);

    int64_t cur_src = 0;
    int64_t cur_dst = 0;
    while(graph->ecount < ecount && !gzeof(f))
    {
        int64_t dif_src;
        int64_t dif_dst;

        UNUSED int len = gzread(f, &dif_src, sizeof(dif_src)) + gzread(f, &dif_dst, sizeof(dif_dst));
        assert(len == sizeof(dif_src) + sizeof(dif_dst));

        cur_src += dif_src;
        cur_dst += dif_dst;
        graph->efr[graph->ecount] = (graph_size_t) cur_src;
        graph->eto[graph->ecount] = (graph_size_t) cur_dst;

        if (E_GRAPH_FLAG_TAG_E & graph->flags)
        {
            uint64_t tag; gzread(f, &tag, sizeof(tag));
            graph->etag[graph->ecount] = (graph_size_t) tag;
        }

        graph->ecount++;
    }

    gzclose(f);

    assert(graph->ecount == ecount);
    return graph;
}

eli_graph_t *eli_graph(_read)(FILE *stream, const graph_flags_enum_t flags)
{
    assert(stream != NULL);
    assert(!(E_GRAPH_FLAG_VAL_E & flags));

    eli_graph_t *graph = eli_graph(_new)((graph_flags_enum_t) (flags & ~E_GRAPH_FLAG_SORT));
    eli_graph(_grow)(graph, (graph_size_t) line_count(stream));

    char  *buf_txt = NULL;
    size_t buf_len = 0;
    ssize_t line_len;

    const char *const buf_fmt = (E_GRAPH_FLAG_TAG_E & flags) ? "%zu %zu %zu" : "%zu %zu";

    while ((line_len = getline(&buf_txt, &buf_len, stream)) != -1)
    {
        if (graph->ecount >= graph->esize)
            eli_graph(_grow)(graph, 1);

        if (line_len > 3 && sscanf(buf_txt, buf_fmt, &graph->efr[graph->ecount], &graph->eto[graph->ecount], &graph->etag[graph->ecount]) >= 2)
            ++graph->ecount;
        else if (buf_txt == NULL || (buf_txt[0] != '#' && buf_txt[0] != '%'))
        {
            // Invalid input
            eli_graph(_free)(graph);
            graph = NULL;
            break;
        }
    }

    free(buf_txt);

    if (graph != NULL)
    {
        eli_graph(_shrink)(graph);
        eli_graph(_toggle_flag)(graph, E_GRAPH_FLAG_SORT, E_GRAPH_FLAG_SORT & flags);
    }

    return graph;
}

eli_graph_t *eli_graph(_read_file)(const char *const filename, const graph_flags_enum_t flags, const char *const force_ext)
{
    const char *ext = (force_ext != NULL) ? force_ext : file_extension(filename);
    if (!ext)
        return NULL;

    eli_graph_t *graph = NULL;
    if (strcasecmp(ext, "el")  == 0 || strcasecmp(ext, "el_gz")  == 0 ||
        strcasecmp(ext, "eli") == 0 || strcasecmp(ext, "eli_gz") == 0 ||
        strcasecmp(ext, "del") == 0 || strcasecmp(ext, "del_gz") == 0 ||
        strcasecmp(ext, "uel") == 0 || strcasecmp(ext, "uel_gz") == 0)
    {
        FILE* file = (filename == NULL || strcmp(filename, "-") == 0)
            ? stdin
            : fopen(filename, "r");

        if (file)
        {
            if (strlen(ext) > 3 && strcmp(ext + strlen(ext) - 3, "_gz") == 0)
                graph = eli_graph(_deserialize)(file);
            else
                graph = eli_graph(_read)(file, flags);

            if (file != stdin)
                fclose(file);

            eli_graph(_toggle_flag)(graph, E_GRAPH_FLAG_PIN,  E_GRAPH_FLAG_PIN & flags);
            eli_graph(_toggle_flag)(graph, E_GRAPH_FLAG_SORT, E_GRAPH_FLAG_SORT & flags);
            assert(graph->flags == flags);
        }

        if (graph != NULL && (strcasecmp(ext, "uel") == 0 || strcasecmp(ext, "uel_gz") == 0))
            eli_graph(_to_directed)(graph);
    }

    return graph;
}

eli_graph_t *eli_graph(_copy)(const eli_graph_t *base, graph_flags_enum_t flags)
{
    assert(base != NULL);

    flags = copy_graph_flags(base->flags, flags);
    eli_graph_t *copy = eli_graph(_new)(flags);
    eli_graph(_set_size)(copy, base->esize);

    if (!(E_GRAPH_FLAG_TAG_E & flags))
        copy->etag = base->etag;

    #ifdef GRAPH_E_TYPE
        if (!(E_GRAPH_FLAG_VAL_E & flags))
            copy->eval = base->eval;
    #endif

    if (base->ecount > 0)
    {
        const graph_size_t ec = base->ecount;
        copy->ecount = ec;

        memcpy(copy->efr, base->efr, sizeof(*base->efr) * ec);
        memcpy(copy->eto, base->eto, sizeof(*base->eto) * ec);

        if (E_GRAPH_FLAG_TAG_E & flags)
            memcpy(copy->etag, base->etag, sizeof(*base->etag) * ec);

        #ifdef GRAPH_E_TYPE
            if (E_GRAPH_FLAG_VAL_E & flags)
                memcpy(copy->eval, base->eval, sizeof(*base->eval) * ec);
        #endif
    }

    return copy;
}

eli_graph_t *eli_graph(_mapped_copy)(const eli_graph_t *base, graph_flags_enum_t flags, const graph_map_edge_func_t map_edge, void *map_arg)
{
    assert(base != NULL);
    if (map_edge == NULL)
        return eli_graph(_copy)(base, flags);

    flags = copy_graph_flags(base->flags, (graph_flags_enum_t) (flags & ~E_GRAPH_FLAG_SORT));
    eli_graph_t *copy = eli_graph(_new)(flags);
    eli_graph(_grow)(copy, 1);

    graph_size_t ecount = 0;

    eli_forall_edges(e, src, dst, base)
    {
        if (map_edge(src, dst, &copy->efr[ecount], &copy->eto[ecount], map_arg))
        {
            if (E_GRAPH_FLAG_TAG_E & flags)
                copy->etag[ecount] = base->etag[e];

            #ifdef GRAPH_E_TYPE
                if (E_GRAPH_FLAG_VAL_E & flags)
                    copy->eval[ecount] = base->eval[e];
            #endif

            eli_graph(_grow)(copy, 1);
            copy->ecount = ++ecount;
        }
    }

    eli_graph(_toggle_flag)(copy, E_GRAPH_FLAG_SORT, E_GRAPH_FLAG_SORT & flags);
    return copy;
}

void eli_graph(_free)(eli_graph_t *graph)
{
    assert(graph != NULL);
    const bool pinned = E_GRAPH_FLAG_PIN & graph->flags;

    memory_pinned_free_if(pinned, graph->efr);
    memory_pinned_free_if(pinned, graph->eto);

    if (E_GRAPH_FLAG_TAG_E & graph->flags)
        memory_pinned_free_if(pinned, graph->etag);

    #ifdef GRAPH_E_TYPE
        if (E_GRAPH_FLAG_VAL_E & graph->flags)
            memory_pinned_free_if(pinned, graph->eval);
    #endif

    memory_free((void*)graph);
}

void eli_graph(_clear)(eli_graph_t *graph)
{
    assert(graph != NULL);

    graph->ecount = 0;
    eli_graph(_set_size)(graph,  0);
}

void eli_graph(_clear_tags)(eli_graph_t *graph)
{
    assert(graph != NULL);

    if (graph->etag != NULL)
        memset(graph->etag, 0, sizeof(*graph->etag) * graph->ecount);
}

#if defined(GRAPH_E_TYPE)
    void eli_graph(_clear_values)(eli_graph_t *graph)
    {
        assert(graph != NULL);

        #ifdef GRAPH_E_TYPE
            if (graph->eval != NULL)
                memset(graph->eval, 0, sizeof(*graph->eval) * graph->ecount);
        #endif
    }
#endif

bool eli_graph(_equals)(const eli_graph_t *restrict first, const eli_graph_t *restrict second)
{
    if (first == second)
        return true;

    assert(first != NULL && second != NULL);

    const graph_size_t ecount = first->ecount;

    return (ecount == second->ecount)
        && (first->efr == second->efr || memcmp(first->efr, second->efr, sizeof(*first->efr) * ecount))
        && (first->eto == second->eto || memcmp(first->eto, second->eto, sizeof(*first->eto) * ecount));
}

void eli_graph(_serialize)(const eli_graph_t *graph, FILE *stream)
{
    assert(graph != NULL);
    assert(stream != NULL);

    gzFile f = gzdopen(dup(fileno(stream)), "wb");
    assert(f != NULL);

    uint64_t flags  = (uint64_t) graph->flags;  gzwrite(f, &flags,  sizeof(flags));
    uint64_t ecount = (uint64_t) graph->ecount; gzwrite(f, &ecount, sizeof(ecount));

    int64_t pre_src = 0;
    int64_t pre_dst = 0;
    eli_forall_edges(e, src, dst, graph)
    {
        int64_t dif_src = pre_src; pre_src = (int64_t)src; dif_src = pre_src - dif_src;
        int64_t dif_dst = pre_dst; pre_dst = (int64_t)dst; dif_dst = pre_dst - dif_dst;
        gzwrite(f, &dif_src, sizeof(dif_src));
        gzwrite(f, &dif_dst, sizeof(dif_dst));

        if (E_GRAPH_FLAG_TAG_E & graph->flags)
        {
            const uint64_t tag = (uint64_t) graph->etag[e];
            gzwrite(f, &tag, sizeof(tag));
        }
    }

    gzclose(f);
}

void eli_graph(_write)(const eli_graph_t *graph, FILE *stream)
{
    assert(graph != NULL);
    assert(stream != NULL);

    if (E_GRAPH_FLAG_TAG_E & graph->flags)
        eli_forall_edges(e, src, dst, graph)
        {
            fprintf(stream, "%zu %zu %zu\n", (size_t)src, (size_t)dst, (size_t)graph->etag[e]);
        }
    else
        eli_forall_edges(src, dst, graph)
        {
            fprintf(stream, "%zu %zu\n", (size_t)src, (size_t)dst);
        }
}

bool eli_graph(_write_file)(eli_graph_t *graph, const char *const filename, const char *const force_ext)
{
    assert(graph != NULL);

    const char *ext = (force_ext != NULL) ? force_ext : file_extension(filename);
    if (!ext)
        return false;

    if (strcasecmp(ext, "el")  == 0 || strcasecmp(ext, "el_gz")  == 0 ||
        strcasecmp(ext, "eli") == 0 || strcasecmp(ext, "eli_gz") == 0 ||
        strcasecmp(ext, "del") == 0 || strcasecmp(ext, "del_gz") == 0 ||
        strcasecmp(ext, "uel") == 0 || strcasecmp(ext, "uel_gz") == 0)
    {
        FILE* file = (filename == NULL || strcmp(filename, "-") == 0)
            ? stdout
            : fopen(filename, "w+");

        if (file)
        {
            if (strcasecmp(ext, "uel") == 0 || strcasecmp(ext, "uel_gz") == 0)
                eli_graph(_to_undirected)(graph);
            if (strcasecmp(ext, "del") == 0 || strcasecmp(ext, "del_gz") == 0 ||
                strcasecmp(ext, "uel") == 0 || strcasecmp(ext, "uel_gz") == 0)
            {
                eli_graph(_remove_dup_edges)(graph, graph_merge_tag_add, NULL);
            }

            if(strlen(ext) > 3 && strcmp(ext + strlen(ext) - 3, "_gz") == 0)
                eli_graph(_serialize)(graph, file);
            else
                eli_graph(_write)(graph, file);

            if (file != stdout)
                fclose(file);
            return true;
        }
    }

    return false;
}

size_t eli_graph(_byte_size)(eli_graph_t *graph, bool allocated)
{
    assert(graph != NULL);

    const graph_size_t e = (allocated) ? graph->esize : graph->ecount;

    size_t s = sizeof(*graph)
        + (sizeof(*graph->efr) * e)
        + (sizeof(*graph->eto) * e);

    if (E_GRAPH_FLAG_TAG_E & graph->flags)
        s += sizeof(*graph->etag) * e;

    #ifdef GRAPH_E_TYPE
        if (E_GRAPH_FLAG_VAL_E & graph->flags)
            s += sizeof(*graph->eval) * e;
    #endif

    return s;
}

void eli_graph(_set_size)(eli_graph_t *graph, graph_size_t esize)
{
    assert(graph != NULL);

    const bool         pinned    = E_GRAPH_FLAG_PIN & graph->flags;
    const graph_size_t alignment = memory_get_default_alignment();

    esize = ROUND_TO_MULT(MAX(esize, graph->ecount), alignment);

    if (graph->esize != esize)
    {
        graph->efr = memory_pinned_retalloc_if(pinned, graph->efr, graph->esize, esize);
        graph->eto = memory_pinned_retalloc_if(pinned, graph->eto, graph->esize, esize);

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

void eli_graph(_grow)(eli_graph_t *graph, graph_size_t egrow)
{
    assert(graph != NULL);

    egrow = (graph->ecount + egrow <= graph->esize)
         ? graph->esize
         : MAX((graph_size_t)(graph->esize * GRAPH_E_GROW), graph->ecount + egrow);

    eli_graph(_set_size)(graph, egrow);
}

void eli_graph(_shrink)(eli_graph_t *graph)
{
    eli_graph(_set_size)(graph, graph->ecount);
}

bool eli_graph(_toggle_flag)(eli_graph_t *graph, const graph_flags_enum_t flag, const bool enable)
{
    assert(graph != NULL);

    if (((graph->flags & flag) != 0) != enable)
    {
        switch(flag)
        {
            case E_GRAPH_FLAG_PIN:
            {
                const memory_manager_enum_t mm_fr = (enable) ? E_MM_DEFAULT : memory_get_default_pinned_manager();
                const memory_manager_enum_t mm_to = (enable) ? memory_get_default_pinned_manager() : E_MM_DEFAULT;

                #define _toggle_pinned_(ptr,size) do {if (ptr) {ptr = CAST_TO_TYPE_OF(ptr) memory_pinned_realloc_managers(mm_fr, mm_to, ptr, (size) * sizeof(*ptr)); }} while(false)

                _toggle_pinned_(graph->efr, graph->esize);
                _toggle_pinned_(graph->eto, graph->esize);

                _toggle_pinned_(graph->etag, graph->esize);

                #ifdef GRAPH_E_TYPE
                    _toggle_pinned_(graph->eval, graph->esize);
                #endif

                #undef _toggle_pinned_
                break;
            }

            case E_GRAPH_FLAG_SORT:
            {
                if (enable)
                    eli_graph(_sort)(graph);
                break;
            }

            case E_GRAPH_FLAG_TAG_E: memory_pinned_toggle_if(graph->flags & E_GRAPH_FLAG_PIN, enable, graph->etag, graph->esize); break;

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

void eli_graph(_sort_ex)(eli_graph_t *graph, const graph_size_t lo, const graph_size_t hi)
{
    const graph_size_t p_idx = lo + (rand() % (hi - lo));
    const graph_size_t p_efr = graph->efr[p_idx];
    const graph_size_t p_eto = graph->eto[p_idx];

    graph_size_t l = lo;
    graph_size_t h = hi;

    do
    {
        while (graph->efr[l] < p_efr || (graph->efr[l] == p_efr && graph->eto[l] < p_eto)) ++l;
        while (graph->efr[h] > p_efr || (graph->efr[h] == p_efr && graph->eto[h] > p_eto)) --h;

        if (l != h)
        {
            if (l > h)
                break;

            SWAP_VALUES(graph->efr[l], graph->efr[h]);
            SWAP_VALUES(graph->eto[l], graph->eto[h]);

            if (graph->etag != NULL) SWAP_VALUES(graph->etag[l], graph->etag[h]);
            #ifdef GRAPH_E_TYPE
                if (graph->eval != NULL) SWAP_VALUES(graph->eval[l], graph->eval[h]);
            #endif
        }

        if (l < hi) l++;
        if (h > lo) h--;
    } while (l <= h);

    if (l < hi)
    {
        if (hi-l >= OMP_TASK_CUTOFF)
            OMP_PRAGMA(omp task)
            eli_graph(_sort_ex)(graph, l, hi);
        else
            eli_graph(_sort_ex)(graph, l, hi);
    }
    if (h > lo)
    {
        if (h-lo >= OMP_TASK_CUTOFF)
            OMP_PRAGMA(omp task)
            eli_graph(_sort_ex)(graph, lo, h);
        else
            eli_graph(_sort_ex)(graph, lo, h);
    }
}

void eli_graph(_sort)(eli_graph_t *graph)
{
    assert(graph != NULL);
    if (graph->ecount < 2)
        return;

    OMP_PRAGMA(omp parallel)
    {
        OMP_PRAGMA(omp single)
        eli_graph(_sort_ex)(graph, 0, graph->ecount - 1);
    }
}

void eli_graph(_transpose)(eli_graph_t *graph)
{
    assert(graph != NULL);
    SWAP_VALUES(graph->efr, graph->eto);
    if (E_GRAPH_FLAG_SORT & graph->flags)
        eli_graph(_sort)(graph);
}

void eli_graph(_to_directed)(eli_graph_t *graph)
{
    assert(graph != NULL);
    const graph_size_t ecount = graph->ecount;

    if (ecount < 1)
        return;

    eli_graph(_grow)(graph, ecount);
    graph->ecount += ecount;

    memcpy(&graph->efr[ecount], graph->eto, sizeof(*graph->efr) * ecount);
    memcpy(&graph->eto[ecount], graph->efr, sizeof(*graph->eto) * ecount);

    if (graph->etag != NULL)
        memcpy(&graph->etag[ecount], graph->etag, sizeof(*graph->etag) * ecount);

    #ifdef GRAPH_E_TYPE
        if (graph->eval != NULL)
            memcpy(&graph->eval[ecount], graph->eval, sizeof(*graph->eval) * ecount);
    #endif

    if (E_GRAPH_FLAG_SORT & graph->flags)
        eli_graph(_sort)(graph);
}

graph_size_t eli_graph(_to_undirected)(eli_graph_t *graph)
{
    assert(graph != NULL);
    graph_size_t removed = 0;

    eli_forall_edges(e, src, dst, graph)
    {
        if (src > dst)
            removed++;
        else if (removed > 0)
        {
            graph->efr[e - removed] = src;
            graph->eto[e - removed] = dst;
            if (graph->etag != NULL) graph->etag[e - removed] = graph->etag[e];
            #ifdef GRAPH_E_TYPE
                if (graph->eval != NULL) graph->eval[e - removed] = graph->eval[e];
            #endif
        }
    }

    graph->ecount -= removed;
    return removed;
}

graph_size_t eli_graph(_remove_dup_edges)(eli_graph_t *graph, const graph_merge_tag_func_t merge_tag, void *merge_arg)
{
    assert(graph != NULL);
    graph_size_t removed = 0;

    if (!(E_GRAPH_FLAG_SORT & graph->flags))
        eli_graph(_sort)(graph);

    for (graph_size_t e = 1; e < graph->ecount; e++)
        if (graph->eto[e-1] == graph->eto[e] && graph->efr[e-1] == graph->efr[e])
        {
            if (merge_tag != NULL && graph->etag != NULL)
                graph->etag[e-1] = merge_tag(graph->etag[e-1], graph->etag[e], merge_arg);
            removed++;
        }
        else if (removed > 0)
        {
            graph->efr[e - removed] = graph->efr[e];
            graph->eto[e - removed] = graph->eto[e];
            if (graph->etag != NULL) graph->etag[e - removed] = graph->etag[e];
            #ifdef GRAPH_E_TYPE
                if (graph->eval != NULL) graph->eval[e - removed] = graph->eval[e];
            #endif
        }

    graph->ecount -= removed;
    return removed;
}

graph_size_t eli_graph(_remove_self_loops)(eli_graph_t *graph)
{
    assert(graph != NULL);
    graph_size_t removed = 0;

    eli_forall_edges(e, src, dst, graph)
    {
        if (src == dst)
            removed++;
        else if (removed > 0)
        {
            graph->efr[e - removed] = src;
            graph->eto[e - removed] = dst;
            if (graph->etag != NULL) graph->etag[e - removed] = graph->etag[e];
            #ifdef GRAPH_E_TYPE
                if (graph->eval != NULL) graph->eval[e - removed] = graph->eval[e];
            #endif
        }
    }

    graph->ecount -= removed;
    return removed;
}

void eli_graph(_get_vertex_range)(const eli_graph_t *graph, graph_size_t *min, graph_size_t *max)
{
    assert(graph != NULL);
    if (min == NULL && max == NULL)
        return;

    graph_size_t vmin = (graph_size_t)SIZE_MAX;
    graph_size_t vmax = 0;

    if (min != NULL)
        eli_forall_edges_par(e, graph, reduction(min:vmin))
        {
            if (graph->efr[e] < vmin)
                vmin = graph->efr[e];
            if (graph->eto[e] < vmin)
                vmin = graph->eto[e];
        }

    if (max != NULL)
        eli_forall_edges_par(e, graph, reduction(max:vmax))
        {
            if (graph->efr[e] > vmax)
                vmax = graph->efr[e];
            if (graph->eto[e] > vmax)
                vmax = graph->eto[e];
        }

    if (min != NULL) *min = vmin;
    if (max != NULL) *max = vmax;
}

void eli_graph(_shift_vertex_range)(eli_graph_t *graph, const int shift)
{
    assert(graph != NULL);
    if (shift == 0)
        return;

    eli_forall_edges_par(e, graph,  /*no omp params*/)
    {
        graph->efr[e] += shift;
        graph->eto[e] += shift;
    }
}

void eli_graph(_rebase_vertex_range)(eli_graph_t *graph, const graph_size_t idx_zero)
{
    assert(graph != NULL);
    graph_size_t vmin = 0;
    eli_graph(_get_vertex_range)(graph, &vmin, NULL);
    eli_graph(_shift_vertex_range)(graph, (int)idx_zero - (int)vmin);
}

bool eli_graph(_get_edge_index)(const eli_graph_t *graph, const graph_size_t src, const graph_size_t dst, graph_size_t *res)
{
    assert(graph != NULL);

    if (E_GRAPH_FLAG_SORT & graph->flags)
    {
        size_t idx = 0;
        if (BINARY_SEARCH(src, graph->efr, 0, graph->ecount - 1, idx))
        {
            graph_size_t lo = idx;
            while (graph->efr[lo-1] == src) lo--;
            graph_size_t hi = idx;
            while (graph->efr[hi+1] == src) hi++;

            return BINARY_SEARCH(dst, graph->eto, lo, hi, *res);
        }

        if (res != NULL) *res = idx;
        return false;
    }
    else
    {
        eli_forall_edges(e, s, d, graph)
        {
            if (s == src && d == dst)
            {
                if (res != NULL) *res = e;
                return true;
            }
        }

        if (res != NULL) *res = graph->ecount;
        return false;
    }
}

graph_size_t eli_graph(_add_edges)(eli_graph_t *graph, const graph_size_t src, const graph_size_t dst, const graph_size_t count)
{
    graph_size_t idx;
    eli_graph(_get_edge_index)(graph, src, dst, &idx);

    if (count > 0)
    {
        const graph_size_t len = graph->ecount;
        if (idx > len)
            idx = len;

        eli_graph(_grow)(graph, count);
        graph->ecount += count;

        // Move old indices to create space
        memmove(&graph->efr[idx+count], &graph->efr[idx], sizeof(*graph->efr) * (len - idx));
        memmove(&graph->eto[idx+count], &graph->eto[idx], sizeof(*graph->eto) * (len - idx));

        // Initialize new edges' destinations
        for (graph_size_t e = 0; e < count; e++)
        {
            graph->efr[idx + e] = src;
            graph->eto[idx + e] = dst;
        }

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
    }

    return idx;
}

graph_size_t eli_graph(_add_edge)(eli_graph_t *graph, const graph_size_t src, const graph_size_t dst)
{
    return eli_graph(_add_edges)(graph, src, dst, 1);
}

void eli_graph(_add_edgelist)(eli_graph_t *graph, const graph_size_t *src, const graph_size_t *dst, const graph_size_t count)
{
    eli_graph(_grow)(graph, count);
    memcpy(&graph->efr[graph->ecount], src, sizeof(*graph->efr) * count);
    memcpy(&graph->eto[graph->ecount], dst, sizeof(*graph->eto) * count);
    graph->ecount += count;

    if (E_GRAPH_FLAG_SORT & graph->flags)
        eli_graph(_sort)(graph);
}
