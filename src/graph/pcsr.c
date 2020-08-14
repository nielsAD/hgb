// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/pcsr.h"
#include "util/file.h"

#define GRAPH_INCLUDE_FILE "graph/template/pcsr.c"
#include "graph/include_all_types.h"
#undef GRAPH_INCLUDE_FILE

#define PCSR_GRAPH_FILENAME_FMT_BASE "%.*s"
#define PCSR_GRAPH_FILENAME_FMT_EXT  "%s"
#define PCSR_GRAPH_FILENAME_VAR_BASE (int)_len, _base
#define PCSR_GRAPH_FILENAME_VAR_EXT  (_ext ? _ext - 1 : NULL)

#define PCSR_GRAPH_FILENAME_FMT(base, fmt, ...) ({\
    assert(base != NULL); \
    DECLARE_TYPE_OF(base) _base = (base); \
    const char         *_ext = file_extension(_base); \
    const graph_size_t  _len = strlen(_base) - (_ext == NULL ? 0 : strlen(_ext) + 1); \
    char *_str; \
    asprintf(&_str, fmt, ##__VA_ARGS__) != -1 ? _str : NULL; \
})

char *pcsr_graph_filename_index(const char *const base, const graph_size_t num_parts)
{
    return PCSR_GRAPH_FILENAME_FMT(base,
           PCSR_GRAPH_FILENAME_FMT_BASE ".p%zu.index" ,
           PCSR_GRAPH_FILENAME_VAR_BASE, (size_t)num_parts);
}

char *pcsr_graph_filename_crossgraph(const char *const base, const graph_size_t num_parts)
{
    return PCSR_GRAPH_FILENAME_FMT(base,
           PCSR_GRAPH_FILENAME_FMT_BASE ".p%zu.cross",
           PCSR_GRAPH_FILENAME_VAR_BASE, (size_t)num_parts);
}

char *pcsr_graph_filename_partition(const char *const base, const graph_size_t num_parts, const graph_size_t part)
{
    return PCSR_GRAPH_FILENAME_FMT(base,
           PCSR_GRAPH_FILENAME_FMT_BASE ".p%zu_%zu"                       PCSR_GRAPH_FILENAME_FMT_EXT,
           PCSR_GRAPH_FILENAME_VAR_BASE, (size_t)num_parts, (size_t)part, PCSR_GRAPH_FILENAME_VAR_EXT);
}

graph_size_t pcsr_global_graph_vcount(const graph_size_t num_parts, const char *const base)
{
    assert(num_parts > 0);

    char *name = pcsr_graph_filename_index(base, num_parts);
    graph_size_t res = 0;

    FILE *f = fopen(name, "r");
    if (f)
    {
        res = line_count(f);
        fclose(f);
    }

    free(name);
    return res;
}

graph_size_t pcsr_cross_graph_tag_encode_parts(const graph_size_t fr, const graph_size_t to)
{
    assert(sizeof(graph_size_t) >= 4);
    assert(fr <= PCSR_GRAPH_MAX_PCOUNT && fr < 0xFFFF);
    assert(to <= PCSR_GRAPH_MAX_PCOUNT && to < 0xFFFF);
    return (fr << 16) | to;
}

void pcsr_cross_graph_tag_decode_parts(const graph_size_t tag, graph_size_t *fr, graph_size_t *to)
{
    if (fr != NULL) *fr = (tag & 0xFFFF0000) >> 16;
    if (to != NULL) *to =  tag & 0xFFFF;
}

graph_size_t pcsr_cross_graph_tag_transpose_parts(const graph_size_t tag)
{
    graph_size_t fr, to;
    pcsr_cross_graph_tag_decode_parts(tag, &fr, &to);
    return pcsr_cross_graph_tag_encode_parts(to, fr);
}

pcsr_cross_graph_t *pcsr_cross_graph_from_eli(const pcsr_cross_graph_eli_t *restrict base, const graph_size_t part, const bool transposed)
{
    assert(base != NULL);
    assert((base->flags & PCSR_CROSS_FLAGS_DEFAULT) == (PCSR_CROSS_FLAGS_DEFAULT & ELI_VALID_FLAGS_MASK));

    pcsr_cross_graph_eli_t *eli = pcsr_cross_graph_eli(_copy)(base, base->flags);

    if (transposed)
    {
        pcsr_cross_graph_eli(_toggle_flag)(eli, E_GRAPH_FLAG_SORT, false);
        pcsr_cross_graph_transpose_eli(eli);
    }

    if (!(E_GRAPH_FLAG_SORT & eli->flags))
        pcsr_cross_graph_sort_eli(eli);

    eli_forall_edges_par(e, eli, /*no omp params*/)
    {
        const graph_size_t src = eli->efr[e];
        const graph_size_t dst = eli->eto[e];
        pcsr_cross_graph_tag_decode_parts(eli->etag[e], &eli->efr[e], &eli->eto[e]);

        eli->etag[e] = (eli->efr[e] == part) ? src : dst;
    }

    pcsr_cross_graph_t *res = pcsr_cross_graph(_convert_from_eli)(eli, (graph_flags_enum_t)(eli->flags | (PCSR_CROSS_FLAGS_DEFAULT & ~ELI_VALID_FLAGS_MASK)));
    pcsr_cross_graph_eli(_free)(eli);

    return res;
}

void pcsr_cross_graph_transpose_eli(pcsr_cross_graph_eli_t *restrict eli)
{
    assert(eli != NULL);
    pcsr_cross_graph_eli(_transpose)(eli);

    eli_forall_edges_par(e, eli, /*no omp params*/)
    {
        eli->etag[e] = pcsr_cross_graph_tag_transpose_parts(eli->etag[e]);
    }
}

void pcsr_cross_graph_sort_eli(pcsr_cross_graph_eli_t *restrict eli)
{
    assert(eli != NULL);

    SWAP_VALUES(eli->efr, eli->eto)
    SWAP_VALUES(eli->efr, eli->etag)
    pcsr_cross_graph_eli(_sort)(eli);
    eli->flags = (graph_flags_enum_t)(eli->flags | E_GRAPH_FLAG_SORT);
    SWAP_VALUES(eli->efr, eli->etag)
    SWAP_VALUES(eli->efr, eli->eto)
}

pcsr_cross_graph_t *pcsr_cross_graph_read_file(const graph_size_t part, const graph_size_t num_parts, const bool transposed, const graph_flags_enum_t flags, const char *const base)
{
    char *name = pcsr_graph_filename_crossgraph(base, num_parts);
    pcsr_cross_graph_eli_t *eli = pcsr_cross_graph_eli(_read_file)(name, (graph_flags_enum_t) (PCSR_CROSS_FLAGS_INHERIT(flags) & ELI_VALID_FLAGS_MASK), "el_gz");

    free(name);
    if (eli == NULL)
        return NULL;

    eli->flags = (graph_flags_enum_t)(eli->flags | E_GRAPH_FLAG_SORT);

    pcsr_cross_graph_t *res = pcsr_cross_graph_from_eli(eli, part, transposed);
    pcsr_cross_graph_eli(_free)(eli);

    return res;
}

static bool _pcsr_cross_filter_mapping_fun_e(const graph_size_t old_src, const graph_size_t old_dst, graph_size_t *restrict new_src, graph_size_t *restrict new_dst, graph_size_t *restrict part)
{
    assert(new_src != NULL);
    assert(new_dst != NULL);
    assert(part != NULL);

    if ((old_src != *part) && (old_dst != *part))
        return false;

    *new_src = old_src;
    *new_dst = old_dst;

    return true;
}
