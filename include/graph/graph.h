// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"

#define GRAPH_BASENAME graph

typedef enum graph_flags {
    E_GRAPH_FLAG_NONE  = 0,
    E_GRAPH_FLAG_PIN   = 1 << 0,
    E_GRAPH_FLAG_SORT  = 1 << 1,
    E_GRAPH_FLAG_TAG_V = 1 << 2,
    E_GRAPH_FLAG_TAG_E = 1 << 3,
    E_GRAPH_FLAG_VAL_V = 1 << 4,
    E_GRAPH_FLAG_VAL_E = 1 << 5,
    E_GRAPH_FLAG_DEG_I = 1 << 6,
    E_GRAPH_FLAG_DEG_O = 1 << 7,
    E_GRAPH_FLAG_DEG_IO = E_GRAPH_FLAG_DEG_I | E_GRAPH_FLAG_DEG_O
} graph_flags_enum_t;

static const graph_flags_enum_t GRAPH_VALID_FLAGS_MASK = (graph_flags_enum_t) ((1 << 8) -1);

typedef unsigned int graph_size_t;
typedef unsigned int graph_tag_t;

typedef bool (*graph_map_vertex_func_t)(const graph_size_t old_index, graph_size_t *new_index, void *arg);
typedef bool (*graph_map_edge_func_t)(const graph_size_t old_src, const graph_size_t old_dst, graph_size_t *restrict new_src, graph_size_t *restrict new_dst, void *arg);
typedef graph_tag_t (*graph_merge_tag_func_t)(const graph_tag_t tag_a, const graph_tag_t tag_b, void *arg);

extern float GRAPH_V_GROW;
extern float GRAPH_E_GROW;

static inline graph_flags_enum_t copy_graph_flags(const graph_flags_enum_t base, const graph_flags_enum_t flags)
{
    // E_GRAPH_FLAG_PIN can always be "copied"
    return (graph_flags_enum_t) ((base | E_GRAPH_FLAG_PIN) & flags);
}

static inline bool graph_map_vertex_noop(const graph_size_t old_index, graph_size_t *new_index, UNUSED void *arg)
{
    assert(new_index != NULL);
    *new_index = old_index;
    return true;
}

static inline bool graph_map_edge_noop(const graph_size_t old_src, const graph_size_t old_dst, graph_size_t *restrict new_src, graph_size_t *restrict new_dst, UNUSED void *arg)
{
    assert(new_src != NULL);
    assert(new_dst != NULL);
    *new_src = old_src;
    *new_dst = old_dst;
    return true;
}

static inline graph_tag_t graph_merge_tag_add(const graph_tag_t tag_a, const graph_tag_t tag_b, UNUSED void *arg)
{
    return tag_a + tag_b;
}
