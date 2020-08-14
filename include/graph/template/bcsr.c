// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/bcsr.h"
#include "util/math.h"
#include "util/memory.h"

#ifndef GRAPH_NAME
    #include "graph/template/bcsr.h"
#endif

struct _bcsr_from_csr_mapping_args;
static bool _bcsr_from_csr_mapping_fun_v(const graph_size_t old_index, graph_size_t *restrict new_index, struct _bcsr_from_csr_mapping_args *restrict args);
static bool _bcsr_from_csr_mapping_fun_e(const graph_size_t old_src, const graph_size_t old_dst, graph_size_t *restrict new_src, graph_size_t *restrict new_dst, struct _bcsr_from_csr_mapping_args *restrict args);

bcsr_graph_t *bcsr_graph(_new_ex)(csr_graph_t *restrict *blocks, const graph_size_t bcount, const graph_flags_enum_t flags)
{
    assert(bcount > 0);
    assert(bcount <= BCSR_GRAPH_MAX_BCOUNT);
    assert(!(~BCSR_VALID_FLAGS_MASK & flags));

    bcsr_graph_t *graph = memory_talloc(bcsr_graph_t);
    assert(graph != NULL);

    graph->flags  = flags;
    graph->bcount = bcount;

    graph->blocks_diag = memory_talloc(*graph->blocks_diag, bcount);

    if (blocks != NULL)
        graph->blocks = blocks;
    else
    {
        graph->blocks = memory_talloc(*graph->blocks, bcount, bcount);

        bcsr_forall_blocks_par(row, col, graph, /*no omp params*/)
        {
            graph_flags_enum_t f = flags;

            // Only the diagonal manages memory for vertex value / degrees
            if (col != row)
                f = (graph_flags_enum_t) (f & BCSR_NONDIAG_VALID_FLAGS_MASK);

            graph->blocks[row * bcount + col] = csr_graph(_new)(f);
        }
    }

    bcsr_graph(_update_block_pointers)(graph);
    return graph;
}

bcsr_graph_t *bcsr_graph(_new)(const graph_size_t bcount, const graph_flags_enum_t flags)
{
    return bcsr_graph(_new_ex)(NULL, bcount, flags);
}

bcsr_graph_t *bcsr_graph(_new_random)(const graph_size_t bcount, graph_size_t vcount, graph_size_t ecount, const graph_flags_enum_t flags)
{
    bcsr_graph_t *graph = bcsr_graph(_new)(bcount, flags);

    vcount = ROUND_TO_MULT(vcount, bcount);
    ecount = ROUND_TO_MULT(ecount, bcount * bcount);

    graph->vcount = vcount;

    const unsigned long long int max_vcount = (unsigned long long int)vcount * vcount;
    if (ecount > max_vcount)
        ecount = max_vcount;

    vcount /= bcount;
    ecount /= bcount * bcount;

    bcsr_forall_blocks_par(block, graph, /*no omp params*/)
    {
        const graph_flags_enum_t bf = graph->blocks[block]->flags;
        csr_graph(_free)(graph->blocks[block]);

        graph->blocks[block] = csr_graph(_new_random)(vcount, ecount, bf);

        OMP_PRAGMA(omp atomic)
        graph->ecount += graph->blocks[block]->ecount;
    }

    bcsr_graph(_update_block_degrees)(graph);
    return graph;
}

bcsr_graph_t *bcsr_graph(_read)(FILE *stream, const graph_flags_enum_t flags, const graph_size_t bcount)
{
    csr_graph_t *csr = csr_graph(_read)(stream, flags);
    if (csr == NULL)
        return NULL;

    bcsr_graph_t *bcsr = bcsr_graph(_copy_from_csr)(csr, flags, bcount);
    csr_graph(_free)(csr);

    return bcsr;
}

bcsr_graph_t *bcsr_graph(_read_file)(const char *const filename, const graph_flags_enum_t flags, const graph_size_t bcount, const char *const force_ext)
{
    csr_graph_t *csr = csr_graph(_read_file)(filename, flags, force_ext);
    if (csr == NULL)
        return NULL;

    bcsr_graph_t *bcsr = bcsr_graph(_copy_from_csr)(csr, flags, bcount);
    csr_graph(_free)(csr);

    return bcsr;
}

bcsr_graph_t *bcsr_graph(_copy)(const bcsr_graph_t *base, graph_flags_enum_t flags)
{
    assert(base != NULL);

    flags = copy_graph_flags(base->flags, flags);
    bcsr_graph_t *copy = bcsr_graph(_new)(base->bcount, flags);

    copy->vcount = base->vcount;
    copy->ecount = base->ecount;

    bcsr_forall_blocks_par(block, copy, /*no omp params*/)
    {
        csr_graph(_free)(copy->blocks[block]);
        copy->blocks[block] = csr_graph(_copy)(base->blocks[block], flags);
    }

    bcsr_graph(_update_block_pointers)(copy);
    return copy;
}

bcsr_graph_t *bcsr_graph(_copy_from_csr)(csr_graph_t *base, graph_flags_enum_t flags, const graph_size_t bcount)
{
    assert(base != NULL);

    flags = copy_graph_flags(base->flags, flags);
    bcsr_graph_t *copy = bcsr_graph(_new)(bcount, flags);

    copy->vcount = base->vcount;
    copy->ecount = base->ecount;

    graph_size_t vcount[bcount];
    memset(&vcount, 0, sizeof(vcount));

    //TODO: should be able to calculate this..
    csr_forall_vertices(v, base)
    {
        vcount[vertex_to_block_id(v, bcount)]++;
    }

    graph_size_t* deg_o[bcount][bcount];
    bcsr_forall_blocks(row, col, copy)
    {
        deg_o[row][col] = (graph_size_t*) memory_calloc(sizeof(graph_size_t) * vcount[row]);
    }

    csr_forall_vertices(v, base)
    {
        const graph_size_t vid = vertex_to_block_index(v, bcount);
        const graph_size_t row = vertex_to_block_id(v, bcount);

        csr_forall_out_edges(dst,v,base)
        {
            deg_o[row][vertex_to_block_id(dst, bcount)][vid]++;
        }
    }

    bcsr_forall_blocks_par(row, col, copy, /*no omp params*/)
    {
        const graph_size_t       block = row*bcount + col;
        const graph_flags_enum_t bflag = copy_graph_flags(copy->blocks[block]->flags, flags);

        struct _bcsr_from_csr_mapping_args args;
        args.bcount = bcount;
        args.row = row;
        args.col = col;

        csr_graph(_free)(copy->blocks[block]);

        copy->blocks[block] = csr_graph(_mapped_copy_with_deg)(
            base,
            vcount[row],
            deg_o[row][col],
            bflag,
            (graph_map_vertex_func_t)_bcsr_from_csr_mapping_fun_v,
            (graph_map_edge_func_t)_bcsr_from_csr_mapping_fun_e,
            &args
        );
    }

    bcsr_forall_blocks(row, col, copy)
    {
        memory_free(deg_o[row][col]);
    }

    bcsr_graph(_update_block_degrees)(copy);
    return copy;
}

void bcsr_graph(_free)(bcsr_graph_t *graph)
{
    assert(graph != NULL);

    bcsr_forall_blocks(block, graph)
    {
        csr_graph(_free)(graph->blocks[block]);
    }

    memory_free((void*)graph->blocks);
    memory_free((void*)graph->blocks_diag);
    memory_free((void*)graph);
}

void bcsr_graph(_clear)(bcsr_graph_t *graph)
{
    assert(graph != NULL);

    bcsr_forall_blocks(block, graph)
    {
        csr_graph(_clear)(graph->blocks[block]);
    }

    bcsr_graph(_update_block_pointers)(graph);
}

bool bcsr_graph(_equals)(const bcsr_graph_t *restrict first, const bcsr_graph_t *restrict second)
{
    if (first == second)
        return true;

    assert(first != NULL && second != NULL);

    if (first->bcount != second->bcount || first->vcount != second->vcount || first->ecount != second->ecount)
        return false;

    bcsr_forall_blocks(block, first)
    {
        if (!csr_graph(_equals)(first->blocks[block], second->blocks[block]))
            return false;
    }

    return true;
}

void bcsr_graph(_update_block_pointers)(bcsr_graph_t *graph)
{
    assert(graph != NULL);

    bcsr_forall_diag_blocks(block, bidx, graph)
    {
        graph->blocks_diag[block] = graph->blocks[bidx];
    }

    bcsr_forall_blocks(idx, row, col, graph)
    {
        graph->blocks[idx]->deg_i = graph->blocks_diag[col]->deg_i;
        graph->blocks[idx]->deg_o = graph->blocks_diag[row]->deg_o;

        graph->blocks[idx]->vtag = graph->blocks_diag[row]->vtag;

        #ifdef GRAPH_V_TYPE
            graph->blocks[idx]->vval = graph->blocks_diag[row]->vval;
        #endif
    }
}

void bcsr_graph(_update_block_degrees)(bcsr_graph_t *graph)
{
    assert(graph != NULL);

    bcsr_graph(_update_block_pointers)(graph);

    const graph_size_t bcount = graph->bcount;

    if (E_GRAPH_FLAG_DEG_I & graph->flags)
        bcsr_forall_blocks(row, col, graph)
        {
            if (row != col)
                csr_graph(_calc_vertex_degrees_in)(
                    graph->blocks[row * bcount + col],
                    graph->blocks[row * bcount + col]->deg_i
                );
        }

    if (E_GRAPH_FLAG_DEG_O & graph->flags)
        bcsr_forall_blocks(row, col, graph)
        {
            if (row != col)
                csr_graph(_calc_vertex_degrees_out)(
                    graph->blocks[row * bcount + col],
                    graph->blocks[row * bcount + col]->deg_o
                );
        }
}

size_t bcsr_graph(_byte_size)(bcsr_graph_t *graph, bool allocated)
{
    assert(graph != NULL);
    const graph_size_t bcount = graph->bcount;

    size_t s = sizeof(*graph) + (sizeof(*graph->blocks) * bcount * bcount) + (sizeof(*graph->blocks_diag) * bcount);

    bcsr_forall_blocks(block, graph)
    {
        s += csr_graph(_byte_size)(graph->blocks[block], allocated);
    }

    return s;
}

graph_size_t bcsr_graph(_align_edges)(bcsr_graph_t *graph, const graph_size_t alignment, const graph_size_t dst)
{
    assert(graph != NULL);

    graph_size_t added = 0;

    bcsr_forall_blocks_par(block, graph, reduction(+:added))
    {
        added += csr_graph(_align_edges)(graph->blocks[block], alignment, dst);
    }

    graph->ecount += added;
    bcsr_graph(_update_block_pointers)(graph);

    return added;
}

void bcsr_graph(_set_size)(bcsr_graph_t *graph, graph_size_t vsize)
{
    assert(graph != NULL);

    const graph_size_t bcount = graph->bcount;
    vsize = ROUND_TO_MULT(vsize, bcount) / bcount;

    bcsr_forall_blocks_par(block, graph, /*no omp params*/)
    {
        csr_graph(_set_size)(graph->blocks[block], vsize, graph->blocks[block]->esize);
    }

    bcsr_graph(_update_block_pointers)(graph);
}

void bcsr_graph(_grow)(bcsr_graph_t *graph, graph_size_t vgrow)
{
    assert(graph != NULL);

    const graph_size_t bcount = graph->bcount;
    vgrow = ROUND_TO_MULT(vgrow, bcount) / bcount;

    bcsr_forall_blocks_par(block, graph, /*no omp params*/)
    {
        csr_graph(_grow)(graph->blocks[block], vgrow, 0);
    }

    bcsr_graph(_update_block_pointers)(graph);
}

void bcsr_graph(_shrink)(bcsr_graph_t *graph)
{
    assert(graph != NULL);

    bcsr_forall_blocks_par(block, graph, /*no omp params*/)
    {
        csr_graph(_shrink)(graph->blocks[block]);
    }

    bcsr_graph(_update_block_pointers)(graph);
}

void bcsr_graph(_transpose)(bcsr_graph_t *graph)
{
    assert(graph != NULL);

    bcsr_forall_blocks(block, graph)
    {
        csr_graph(_transpose)(graph->blocks[block]);
    }

    bcsr_forall_blocks(row, col, graph)
    {
        SWAP_VALUES(graph->blocks[row*graph->bcount + col], graph->blocks[col*graph->bcount + row])
    }

    bcsr_graph(_update_block_degrees)(graph);
}

bool bcsr_graph(_toggle_flag)(bcsr_graph_t *graph, const graph_flags_enum_t flag, const bool enable)
{
    assert(graph != NULL);

    if ((graph->flags & flag) != enable)
    {
        bcsr_forall_blocks(idx, row, col, graph)
        {
            const graph_flags_enum_t f = (row == col)
                ? flag
                : (graph_flags_enum_t) (flag & BCSR_NONDIAG_VALID_FLAGS_MASK);

            if (f && !csr_graph(_toggle_flag)(graph->blocks[idx], f, enable))
                return false;
        }

        if (enable)
            graph->flags = (graph_flags_enum_t) (graph->flags | flag);
        else
            graph->flags = (graph_flags_enum_t) (graph->flags & ~flag);

        bcsr_graph(_update_block_pointers)(graph);
        if (enable && (flag & (E_GRAPH_FLAG_DEG_I | E_GRAPH_FLAG_DEG_O)))
            bcsr_graph(_update_block_degrees)(graph);
    }

    return true;
}

bool bcsr_graph(_get_vertex_index)(const bcsr_graph_t *graph, const graph_size_t idx, graph_size_t *bid, graph_size_t *bidx)
{
    assert(graph != NULL);

    const graph_size_t block_id  = vertex_to_block_id(idx, graph->bcount);
    const graph_size_t block_idx = vertex_to_block_index(idx, graph->bcount);

    if (bid  != NULL) *bid  = block_id;
    if (bidx != NULL) *bidx = block_idx;

    return block_idx < graph->blocks_diag[block_id]->vcount;
}

bool bcsr_graph(_get_edge_index)(const bcsr_graph_t *graph, const graph_size_t src, const graph_size_t dst, graph_size_t *block, graph_size_t *idx)
{
    graph_size_t block_src;
    graph_size_t block_dst;
    graph_size_t index_src;
    graph_size_t index_dst;
    if (bcsr_graph(_get_vertex_index)(graph, src, &block_src, &index_src) &&
        bcsr_graph(_get_vertex_index)(graph, dst, &block_dst, &index_dst))
    {
        const graph_size_t block_edge = block_src * graph->bcount + block_dst;
        if (block != NULL)
            *block = block_edge;

        return csr_graph(_get_edge_index)(
                   graph->blocks[block_edge],
                   index_src,
                   index_dst,
                   idx
               );
    }

    return false;
}

void bcsr_graph(_insert_vertices)(bcsr_graph_t *graph, graph_size_t idx, const graph_size_t count)
{
    assert(graph != NULL);

    if (count > 0)
    {
        const graph_size_t bcount = graph->bcount;

        bcsr_graph(_grow)(graph, count);
        graph->vcount += count;

        graph_size_t grow[bcount];
        memset(&grow, 0, sizeof(grow));

        //TODO: should be able to calculate this..
        for (graph_size_t v = 0; v < count; v++)
            grow[vertex_to_block_id(idx + v, bcount)]++;

        bcsr_forall_blocks_par(row, col, graph, /*no omp params*/)
        {
            csr_graph(_insert_vertices)(
                graph->blocks[row * bcount + col],
                vertex_to_block_index(idx, bcount),
                grow[row]
            );
        }
    }
}

void bcsr_graph(_insert_vertex)(bcsr_graph_t *graph, graph_size_t idx)
{
    bcsr_graph(_insert_vertices)(graph, idx, 1);
}

graph_size_t bcsr_graph(_add_vertices)(bcsr_graph_t *graph, const graph_size_t count)
{
    assert(graph != NULL);

    const graph_size_t idx = graph->vcount;
    bcsr_graph(_insert_vertices)(graph, idx, count);

    return idx;
}

graph_size_t bcsr_graph(_add_vertex)(bcsr_graph_t *graph)
{
    return bcsr_graph(_add_vertices)(graph, 1);
}

void bcsr_graph(_add_edges)(bcsr_graph_t *graph, const graph_size_t src, const graph_size_t dst, const graph_size_t count)
{
    if (count > 0)
    {
        graph_size_t block_src;
        graph_size_t block_dst;
        graph_size_t index_src;
        graph_size_t index_dst;
        if (bcsr_graph(_get_vertex_index)(graph, src, &block_src, &index_src) &&
            bcsr_graph(_get_vertex_index)(graph, dst, &block_dst, &index_dst))
        {
            graph->ecount += count;
            csr_graph(_add_edges)(
                graph->blocks[block_src * graph->bcount + block_dst],
                index_src,
                index_dst,
                count
            );
        }
    }
}

void bcsr_graph(_add_edge)(bcsr_graph_t *graph, const graph_size_t src, const graph_size_t dst)
{
    bcsr_graph(_add_edges)(graph, src, dst, 1);
}

void bcsr_graph(_add_edgelist)(bcsr_graph_t *graph, const graph_size_t *src, const graph_size_t *dst, const graph_size_t count)
{
    assert(graph != NULL);

    const graph_size_t bcount = graph->bcount;

    graph_size_t grow[bcount*bcount];
    memset(&grow, 0, sizeof(grow));

    // Count edges in each block
    for (graph_size_t e = 0; e < count; e++)
        grow[vertex_to_block_id(src[e], bcount) * bcount + vertex_to_block_id(dst[e], graph->bcount)]++;

    // Grow blocks
    bcsr_forall_blocks(b, graph)
    {
        csr_graph(_grow)(graph->blocks[b], 0, grow[b]);
    }

    // Add edges
    for (graph_size_t e = 0; e < count; e++)
        bcsr_graph(_add_edge)(graph, src[e], dst[e]);
}

graph_size_t bcsr_graph(_get_vertex_degree)(const bcsr_graph_t *graph, const graph_size_t idx)
{
    return bcsr_graph(_get_vertex_degree_in)(graph, idx) + bcsr_graph(_get_vertex_degree_out)(graph, idx);
}

graph_size_t bcsr_graph(_get_vertex_degree_in)(const bcsr_graph_t *graph, const graph_size_t idx)
{
    graph_size_t bid = 0;
    graph_size_t bidx;
    UNUSED bool res = bcsr_graph(_get_vertex_index)(graph, idx, &bid, &bidx);
    assert(res && "invalid vertex index");

    if (E_GRAPH_FLAG_DEG_I & graph->flags)
        return csr_graph(_get_vertex_degree_in)(graph->blocks_diag[bid], bidx);
    else
    {
        graph_size_t deg = 0;
        bcsr_forall_diag_blocks(row, graph)
        {
            deg += csr_graph(_get_vertex_degree_in)(graph->blocks[row*graph->bcount + bid], bidx);
        }
        return deg;
    }
}

graph_size_t bcsr_graph(_get_vertex_degree_out)(const bcsr_graph_t *graph, const graph_size_t idx)
{
    graph_size_t bid = 0;
    graph_size_t bidx;
    UNUSED bool res = bcsr_graph(_get_vertex_index)(graph, idx, &bid, &bidx);
    assert(res && "invalid vertex index");

    if (E_GRAPH_FLAG_DEG_O & graph->flags)
        return csr_graph(_get_vertex_degree_out)(graph->blocks_diag[bid*graph->bcount], bidx);
    else
    {
        graph_size_t deg = 0;
        bcsr_forall_diag_blocks(col, graph)
        {
            deg += csr_graph(_get_vertex_degree_out)(graph->blocks[bid*graph->bcount + col], bidx);
        }
        return deg;
    }
}
