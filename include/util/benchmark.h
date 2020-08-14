// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"
#include "graph/pcsr.h"
#include "util/string.h"
#include "util/file.h"
#include "util/memory.h"
#include "util/time.h"

#include <argp.h>
#include <wordexp.h>

#define BENCH_MAX_FILE_INPUT 1024

#ifndef BENCH_GRAPH_FLAGS
    #define BENCH_GRAPH_FLAGS E_GRAPH_FLAG_NONE
#endif

#ifndef BENCH_PCSR_PART
    #define BENCH_PCSR_PART 0
#endif

#ifndef BENCH_PCSR_BCOUNT
    #define BENCH_PCSR_BCOUNT 1
#endif

#if (!defined(benchmark_eli_graph)  || \
     !defined(benchmark_csr_graph)  || !defined(benchmark_csc_graph)  || \
     !defined(benchmark_bcsr_graph) || !defined(benchmark_bcsc_graph) ||\
     !defined(benchmark_pcsr_graph) || !defined(benchmark_pcsc_graph))
    #error Undefined template macros, using placeholders for editor

    #define benchmark_eli_graph(ID)  eli_graph_v ## ID
    #define benchmark_csr_graph(ID)  csr_graph_v ## ID
    #define benchmark_csc_graph(ID)  csr_graph_v ## ID
    #define benchmark_bcsr_graph(ID) bcsr_graph_v ## ID
    #define benchmark_bcsc_graph(ID) bcsr_graph_v ## ID
    #define benchmark_pcsr_graph(ID) pcsr_graph_v ## ID
    #define benchmark_pcsc_graph(ID) pcsr_graph_v ## ID

    #define benchmark_eli_graph_t  benchmark_eli_graph(_t)
    #define benchmark_csr_graph_t  benchmark_csr_graph(_t)
    #define benchmark_csc_graph_t  benchmark_csc_graph(_t)
    #define benchmark_bcsr_graph_t benchmark_bcsr_graph(_t)
    #define benchmark_bcsc_graph_t benchmark_bcsc_graph(_t)
    #define benchmark_pcsr_graph_t benchmark_pcsr_graph(_t)
    #define benchmark_pcsc_graph_t benchmark_pcsc_graph(_t)
#endif

typedef struct benchmark_options {
    // benchmark options
    graph_size_t bcount_begin;
    graph_size_t bcount_end;

    size_t start_iterations;
    size_t bench_iterations;

    // graph generator options
    graph_size_t vcount_begin;
    graph_size_t vcount_end;
    graph_size_t ecount_begin;
    graph_size_t ecount_end;

    // benchmark callbacks
    size_t  eli_fun_count;
    size_t  csr_fun_count;
    size_t  csc_fun_count;
    size_t bcsr_fun_count;
    size_t bcsc_fun_count;
    size_t pcsr_fun_count;
    size_t pcsc_fun_count;

    // file inputs
    wordexp_t *files;
} benchmark_options_t;

enum benchmark_cl_options {
    BENCH_OPT_BLCK = 'b',
    BENCH_OPT_VRTX = 'v',
    BENCH_OPT_EDGE = 'e',

    BENCH_OPT_FILE_INPUT = 'f',
    BENCH_OPT_GENERATOR  = 'g',
    BENCH_OPT_RAND_SEED  = 'r',
    BENCH_OPT_MMAP       = 'D',

    BENCH_OPT_ITERATIONS = 'i',
    BENCH_OPT_WARMING_UP = 'w',

    BENCH_OPT_BLCK_BEG = 1,
    BENCH_OPT_BLCK_END,
    BENCH_OPT_VRTX_BEG,
    BENCH_OPT_VRTX_END,
    BENCH_OPT_EDGE_BEG,
    BENCH_OPT_EDGE_END,

    BENCH_OPT_USER
};

#ifndef ARG_BENCH_NUM_BLOCKS_GROW
   #define ARG_BENCH_NUM_BLOCKS_GROW *= 2
#endif

#ifndef ARG_BENCH_NUM_VERTICES_GROW
   #define ARG_BENCH_NUM_VERTICES_GROW *= 4
#endif

#ifndef ARG_BENCH_NUM_EDGES_GROW
   #define ARG_BENCH_NUM_EDGES_GROW *= 4
#endif

#define ARG_BENCH_NUM_BLOCKS \
    {"num_blocks",   BENCH_OPT_BLCK, "int", 0, "Number of blocks.", 0}
#define ARG_BENCH_NUM_VERTICES \
    {"num_vertices", BENCH_OPT_VRTX, "int", 0, "Number of vertices to generate.", 0}
#define ARG_BENCH_NUM_EDGES \
    {"num_edges",    BENCH_OPT_EDGE, "int", 0, "Number of edges to generate.", 0}

#define ARG_BENCH_NUM_BLOCKS_BEG \
    {"num_blocks_begin", BENCH_OPT_BLCK_BEG, "int", 0, "Number of blocks begin loop.", 0},\
    {"bb",               BENCH_OPT_BLCK_BEG, "int", OPTION_ALIAS, NULL, 0}

#define ARG_BENCH_NUM_BLOCKS_END \
    {"num_blocks_end", BENCH_OPT_BLCK_END, "int", 0, "Number of blocks end loop.", 0},\
    {"be",             BENCH_OPT_BLCK_END, "int", OPTION_ALIAS, NULL, 0}

#define ARG_BENCH_NUM_VERTICES_BEG \
    {"num_vertices_begin", BENCH_OPT_VRTX_BEG, "int", 0, "Number of vertices begin loop.", 0},\
    {"vb",                 BENCH_OPT_VRTX_BEG, "int", OPTION_ALIAS, NULL, 0}

#define ARG_BENCH_NUM_VERTICES_END \
    {"num_vertices_end", BENCH_OPT_VRTX_END, "int", 0, "Number of vertices end loop.", 0},\
    {"ve",               BENCH_OPT_VRTX_END, "int", OPTION_ALIAS, NULL, 0}

#define ARG_BENCH_NUM_EDGES_BEG \
    {"num_edges_begin", BENCH_OPT_EDGE_BEG, "int", 0, "Number of edges begin loop.", 0},\
    {"eb",              BENCH_OPT_EDGE_BEG, "int", OPTION_ALIAS, NULL, 0}

#define ARG_BENCH_NUM_EDGES_END \
    {"num_edges_end", BENCH_OPT_EDGE_END, "int", 0, "Number of edges end loop.", 0},\
    {"ee",            BENCH_OPT_EDGE_END, "int", OPTION_ALIAS, NULL, 0}

#define ARG_BENCH_FILE_INPUT \
    {"input", BENCH_OPT_FILE_INPUT, "file", 0, "Input file (edgelist).", 0}

#define ARG_BENCH_GENERATOR \
    {"generator", BENCH_OPT_GENERATOR, "bool", OPTION_ARG_OPTIONAL, "Use random graph generator for input.", 0}

#define ARG_BENCH_RAND_SEED \
    {"random", BENCH_OPT_RAND_SEED, "int", 0, "Random seed.", 0}

#define ARG_BENCH_MMAP \
    {"disk", BENCH_OPT_MMAP, "directory", OPTION_ARG_OPTIONAL, "Specify file template (ending with `XXXXXX`) to store intermediates in temporary files to reduce memory load. Working dir by default.", 0}

#define ARG_BENCH_ITERATIONS \
    {"iterations", BENCH_OPT_ITERATIONS, "int", 0, "Number of loop iterations for each function per benchmark.", 0}

#define ARG_BENCH_WARMING_UP \
    {"warming_up", BENCH_OPT_WARMING_UP, "int", 0, "Numer of warming-up loop iterations before starting benchmark timer.", 0}

#define ARG_BENCH_NUM_BLOCKS_LOOP   ARG_BENCH_NUM_BLOCKS,   ARG_BENCH_NUM_BLOCKS_BEG,   ARG_BENCH_NUM_BLOCKS_END
#define ARG_BENCH_NUM_VERTICES_LOOP ARG_BENCH_NUM_VERTICES, ARG_BENCH_NUM_VERTICES_BEG, ARG_BENCH_NUM_VERTICES_END
#define ARG_BENCH_NUM_EDGES_LOOP    ARG_BENCH_NUM_EDGES,    ARG_BENCH_NUM_EDGES_BEG,    ARG_BENCH_NUM_EDGES_END

#define ARG_BENCH_OPTIONS ARG_BENCH_NUM_BLOCKS_LOOP, ARG_BENCH_MMAP, ARG_BENCH_ITERATIONS, ARG_BENCH_WARMING_UP
#define ARG_BENCH_GENERATOR_INPUT ARG_BENCH_NUM_VERTICES_LOOP, ARG_BENCH_NUM_EDGES_LOOP, ARG_BENCH_GENERATOR, ARG_BENCH_RAND_SEED

static error_t benchmark_argp_parser(int key, char *arg, struct argp_state *state) {
    benchmark_options_t *o = (benchmark_options_t*) state->input;

    switch (key)
    {
        case BENCH_OPT_BLCK: o->bcount_begin = o->bcount_end = strtoullr(arg, o->bcount_begin, o->bcount_end); break;
        case BENCH_OPT_VRTX: o->vcount_begin = o->vcount_end = strtoullr(arg, o->vcount_begin, o->vcount_end); break;
        case BENCH_OPT_EDGE: o->ecount_begin = o->ecount_end = strtoullr(arg, o->ecount_begin, o->ecount_end); break;

        case BENCH_OPT_BLCK_BEG: o->bcount_begin = strtoullr(arg, o->bcount_begin, o->bcount_end); break;
        case BENCH_OPT_BLCK_END: o->bcount_end   = strtoullr(arg, o->bcount_begin, o->bcount_end); break;
        case BENCH_OPT_VRTX_BEG: o->vcount_begin = strtoullr(arg, o->vcount_begin, o->vcount_end); break;
        case BENCH_OPT_VRTX_END: o->vcount_end   = strtoullr(arg, o->vcount_begin, o->vcount_end); break;
        case BENCH_OPT_EDGE_BEG: o->ecount_begin = strtoullr(arg, o->ecount_begin, o->ecount_end); break;
        case BENCH_OPT_EDGE_END: o->ecount_end   = strtoullr(arg, o->ecount_begin, o->ecount_end); break;

        case BENCH_OPT_RAND_SEED: srand((unsigned int)(sstrtoull(arg))); break;
        case BENCH_OPT_GENERATOR:
        {
            if (strtob(arg, true) == false)
            {
                o->vcount_begin = o->vcount_end = 0;
                o->ecount_begin = o->ecount_end = 0;
            }
            break;
        }

        case BENCH_OPT_FILE_INPUT:
        {
            int flags = WRDE_SHOWERR;

            if (o->files == NULL)
                o->files = memory_talloc(*o->files);
            else
                flags |= WRDE_APPEND;

            wordexp(arg, o->files, flags);
            break;
        }

        case BENCH_OPT_MMAP:
        {
            memory_set_default_pinned_manager(E_MM_MMAP);
            memory_set_mapped_tmpdir(arg);
            break;
        }

        case BENCH_OPT_WARMING_UP: o->start_iterations = sstrtoull(arg); break;
        case BENCH_OPT_ITERATIONS: o->bench_iterations = sstrtoull(arg); break;

        default:
            return ARGP_ERR_UNKNOWN;
    }

    return 0;
}

// Callbacks, to be defined by the program
static bool cb_benchmark_before_graph(benchmark_options_t *o, const char *const n);
static void cb_benchmark_after_graph(benchmark_options_t *o);
static bool cb_benchmark_reset_time(benchmark_options_t *o);
static bool cb_benchmark_initialize_par(benchmark_options_t *o, const graph_size_t p);
static bool cb_benchmark_initialize_dist(benchmark_options_t *o, const graph_size_t p);
static void cb_benchmark_finalize_dist(benchmark_options_t *o);

static bool cb_benchmark_initialize_eli (benchmark_options_t *o,  benchmark_eli_graph_t *g);
static bool cb_benchmark_initialize_csr (benchmark_options_t *o,  benchmark_csr_graph_t *g);
static bool cb_benchmark_initialize_csc (benchmark_options_t *o,  benchmark_csc_graph_t *g);
static bool cb_benchmark_initialize_bcsr(benchmark_options_t *o, benchmark_bcsr_graph_t *g);
static bool cb_benchmark_initialize_bcsc(benchmark_options_t *o, benchmark_bcsc_graph_t *g);
static bool cb_benchmark_initialize_pcsr(benchmark_options_t *o, benchmark_pcsr_graph_t *g);
static bool cb_benchmark_initialize_pcsc(benchmark_options_t *o, benchmark_pcsc_graph_t *g);

static bool cb_benchmark_before_eli (benchmark_options_t *o, const  benchmark_eli_graph_t *g, const size_t fun);
static bool cb_benchmark_before_csr (benchmark_options_t *o, const  benchmark_csr_graph_t *g, const size_t fun);
static bool cb_benchmark_before_csc (benchmark_options_t *o, const  benchmark_csc_graph_t *g, const size_t fun);
static bool cb_benchmark_before_bcsr(benchmark_options_t *o, const benchmark_bcsr_graph_t *g, const size_t fun);
static bool cb_benchmark_before_bcsc(benchmark_options_t *o, const benchmark_bcsc_graph_t *g, const size_t fun);
static bool cb_benchmark_before_pcsr(benchmark_options_t *o, const benchmark_pcsr_graph_t *g, const size_t fun);
static bool cb_benchmark_before_pcsc(benchmark_options_t *o, const benchmark_pcsc_graph_t *g, const size_t fun);

static bool cb_benchmark_disp_eli (benchmark_options_t *o, const  benchmark_eli_graph_t *g, const size_t fun);
static bool cb_benchmark_disp_csr (benchmark_options_t *o, const  benchmark_csr_graph_t *g, const size_t fun);
static bool cb_benchmark_disp_csc (benchmark_options_t *o, const  benchmark_csc_graph_t *g, const size_t fun);
static bool cb_benchmark_disp_bcsr(benchmark_options_t *o, const benchmark_bcsr_graph_t *g, const size_t fun);
static bool cb_benchmark_disp_bcsc(benchmark_options_t *o, const benchmark_bcsc_graph_t *g, const size_t fun);
static bool cb_benchmark_disp_pcsr(benchmark_options_t *o, const benchmark_pcsr_graph_t *g, const size_t fun);
static bool cb_benchmark_disp_pcsc(benchmark_options_t *o, const benchmark_pcsc_graph_t *g, const size_t fun);

static bool cb_benchmark_after_eli (benchmark_options_t *o, const  benchmark_eli_graph_t *g, const size_t fun, const double duration);
static bool cb_benchmark_after_csr (benchmark_options_t *o, const  benchmark_csr_graph_t *g, const size_t fun, const double duration);
static bool cb_benchmark_after_csc (benchmark_options_t *o, const  benchmark_csc_graph_t *g, const size_t fun, const double duration);
static bool cb_benchmark_after_bcsr(benchmark_options_t *o, const benchmark_bcsr_graph_t *g, const size_t fun, const double duration);
static bool cb_benchmark_after_bcsc(benchmark_options_t *o, const benchmark_bcsc_graph_t *g, const size_t fun, const double duration);
static bool cb_benchmark_after_pcsr(benchmark_options_t *o, const benchmark_pcsr_graph_t *g, const size_t fun, const double duration);
static bool cb_benchmark_after_pcsc(benchmark_options_t *o, const benchmark_pcsc_graph_t *g, const size_t fun, const double duration);

static void cb_benchmark_finalize_eli (benchmark_options_t *o, benchmark_eli_graph_t *g);
static void cb_benchmark_finalize_csr (benchmark_options_t *o, benchmark_csr_graph_t *g);
static void cb_benchmark_finalize_csc (benchmark_options_t *o, benchmark_csc_graph_t *g);
static void cb_benchmark_finalize_bcsr(benchmark_options_t *o, benchmark_bcsr_graph_t *g);
static void cb_benchmark_finalize_bcsc(benchmark_options_t *o, benchmark_bcsc_graph_t *g);
static void cb_benchmark_finalize_pcsr(benchmark_options_t *o, benchmark_pcsr_graph_t *g);
static void cb_benchmark_finalize_pcsc(benchmark_options_t *o, benchmark_pcsc_graph_t *g);

static bool benchmark_run_graph(benchmark_options_t *o, benchmark_csr_graph_t *base_graph)
{
    assert(o != NULL);
    assert(o->bench_iterations > 0);

    bool success = true;
    #define soc(X)      { if(!(X)) continue; }                          //success or continue
    #define sor_eli(X)  { if(!(X)) {success = false; goto free_eli; }}  //success or free eli and return
    #define sor_csr(X)  { if(!(X)) {success = false; goto free_csr; }}  //success or free csr and return
    #define sor_csc(X)  { if(!(X)) {success = false; goto free_csc; }}  //success or free csc and return
    #define sor_bcsr(X) { if(!(X)) {success = false; goto free_bcsr; }} //success or free bcsr and return
    #define sor_bcsc(X) { if(!(X)) {success = false; goto free_bcsc; }} //success or free bcsc and return

    if (success && o->eli_fun_count > 0)
    {
        benchmark_eli_graph_t *target_graph = benchmark_csr_graph(_get_eli_representation)(base_graph);
        sor_eli(cb_benchmark_initialize_eli(o, target_graph))

        for (graph_size_t p = 0; cb_benchmark_initialize_par(o, p); p++)
            for (size_t d = 0; d < o->csr_fun_count; d++)
            {
                soc(cb_benchmark_before_eli(o, target_graph, d))

                for (uint32_t i = 0; i < o->start_iterations; i++)
                    sor_eli(cb_benchmark_disp_eli(o, target_graph, d))

                if (!cb_benchmark_reset_time(o)) {
                    continue;
                }

                const time_mark_t start_time = time_mark();

                for (size_t i = 0; i < o->bench_iterations; i++)
                    sor_eli(cb_benchmark_disp_eli(o, target_graph, d))

                const time_diff_t duration = time_since(start_time) / o->bench_iterations;

                sor_eli(cb_benchmark_after_eli(o, target_graph, d, duration))
            }

        free_eli:
        cb_benchmark_finalize_eli(o, target_graph);
        benchmark_csr_graph(_free_eli_representation)(base_graph, target_graph);
    }

    if (success && o->csr_fun_count > 0)
    {
        sor_csr(cb_benchmark_initialize_csr(o, base_graph))
        for (graph_size_t p = 0; cb_benchmark_initialize_par(o, p); p++)
            for (size_t d = 0; d < o->csr_fun_count; d++)
            {
                soc(cb_benchmark_before_csr(o, base_graph, d))

                for (size_t i = 0; i < o->start_iterations; i++)
                    sor_csr(cb_benchmark_disp_csr(o, base_graph, d))

                if (!cb_benchmark_reset_time(o)) {
                    continue;
                }

                const time_mark_t start_time = time_mark();

                for (size_t i = 0; i < o->bench_iterations; i++)
                    sor_csr(cb_benchmark_disp_csr(o, base_graph, d))

                const time_diff_t duration = time_since(start_time) / o->bench_iterations;

                sor_csr(cb_benchmark_after_csr(o, base_graph, d, duration))
            }

        free_csr:
        cb_benchmark_finalize_csr(o, base_graph);
    }

    if (success && o->bcsr_fun_count > 0)
        for (graph_size_t bcount = o->bcount_begin; success && bcount <= o->bcount_end; bcount ARG_BENCH_NUM_BLOCKS_GROW)
        {
            benchmark_bcsr_graph_t *target_graph = benchmark_bcsr_graph(_copy_from_csr)(base_graph, base_graph->flags, bcount);
            sor_bcsr(cb_benchmark_initialize_bcsr(o, target_graph))

            for (graph_size_t p = 0; cb_benchmark_initialize_par(o, p); p++)
                for (size_t d = 0; d < o->bcsr_fun_count; d++)
                {
                    soc(cb_benchmark_before_bcsr(o, target_graph, d))

                    for (size_t i = 0; i < o->start_iterations; i++)
                        sor_bcsr(cb_benchmark_disp_bcsr(o, target_graph, d))

                    if (!cb_benchmark_reset_time(o)) {
                        continue;
                    }

                    const time_mark_t start_time = time_mark();

                    for (size_t i = 0; i < o->bench_iterations; i++)
                        sor_bcsr(cb_benchmark_disp_bcsr(o, target_graph, d))

                    const time_diff_t duration = time_since(start_time) / o->bench_iterations;

                    sor_bcsr(cb_benchmark_after_bcsr(o, target_graph, d, duration))
                }

            free_bcsr:
            cb_benchmark_finalize_bcsr(o, target_graph);
            benchmark_bcsr_graph(_free)(target_graph);
        }

    if (success && o->csc_fun_count + o->bcsc_fun_count > 0)
    {
        benchmark_csr_graph(_transpose)(base_graph);

        if (success && o->csc_fun_count > 0)
        {
            sor_csc(cb_benchmark_initialize_csc(o, base_graph))
            for (graph_size_t p = 0; cb_benchmark_initialize_par(o, p); p++)
                for (size_t d = 0; d < o->csc_fun_count; d++)
                {
                    soc(cb_benchmark_before_csc(o, base_graph, d))

                    for (size_t i = 0; i < o->start_iterations; i++)
                        sor_csc(cb_benchmark_disp_csc(o, base_graph, d))

                    if (!cb_benchmark_reset_time(o)) {
                        continue;
                    }

                    const time_mark_t start_time = time_mark();

                    for (size_t i = 0; i < o->bench_iterations; i++)
                        sor_csc(cb_benchmark_disp_csc(o, base_graph, d))

                    const time_diff_t duration = time_since(start_time) / o->bench_iterations;

                    sor_csc(cb_benchmark_after_csc(o, base_graph, d, duration))
                }

            free_csc:
            cb_benchmark_finalize_csc(o, base_graph);
        }

        if (success && o->bcsc_fun_count > 0)
            for (graph_size_t bcount = o->bcount_begin; success && bcount <= o->bcount_end; bcount ARG_BENCH_NUM_BLOCKS_GROW)
            {
                benchmark_bcsc_graph_t *target_graph = benchmark_bcsc_graph(_copy_from_csr)(base_graph, base_graph->flags, bcount);
                sor_bcsc(cb_benchmark_initialize_bcsc(o, target_graph))

                for (graph_size_t p = 0; cb_benchmark_initialize_par(o, p); p++)
                    for (size_t d = 0; d < o->bcsc_fun_count; d++)
                    {
                        soc(cb_benchmark_before_bcsc(o, target_graph, d))

                        for (size_t i = 0; i < o->start_iterations; i++)
                            sor_bcsc(cb_benchmark_disp_bcsc(o, target_graph, d))

                        if (!cb_benchmark_reset_time(o)) {
                            continue;
                        }

                        const time_mark_t start_time = time_mark();

                        for (size_t i = 0; i < o->bench_iterations; i++)
                            sor_bcsc(cb_benchmark_disp_bcsc(o, target_graph, d))

                        const time_diff_t duration = time_since(start_time) / o->bench_iterations;

                        sor_bcsc(cb_benchmark_after_bcsc(o, target_graph, d, duration))
                    }

                free_bcsc:
                cb_benchmark_finalize_bcsc(o, target_graph);
                benchmark_bcsc_graph(_free)(target_graph);
            }

        benchmark_csr_graph(_transpose)(base_graph);
    }

    #undef soc
    #undef sor_eli
    #undef sor_csr
    #undef sor_csc
    #undef sor_bcsr
    #undef sor_bcsc

    return success;
}

static bool benchmark_run_graph_dist(benchmark_options_t *o, const char *const base)
{
    assert(o != NULL);
    assert(o->bench_iterations > 0);

    bool success = true;
    #define soc(X)      { if(!(X)) continue; }                           //success or continue
    #define sor_pcsr(X) { if(!(X)) {success = false; goto free_pcsr; }}  //success or free csr and return
    #define sor_pcsc(X) { if(!(X)) {success = false; goto free_pcsc; }}  //success or free csc and return

    if (success && o->pcsr_fun_count > 0)
        for (graph_size_t pcount = o->bcount_begin; success && pcount <= o->bcount_end; pcount ARG_BENCH_NUM_BLOCKS_GROW)
        {
            soc(cb_benchmark_initialize_dist(o, pcount))

            benchmark_pcsr_graph_t *target_graph = benchmark_pcsr_graph(_read_file)(
                BENCH_PCSR_PART,
                pcount,
                BENCH_PCSR_BCOUNT,
                base,
                (graph_flags_enum_t) BENCH_GRAPH_FLAGS,
                NULL
            );

            if (target_graph == NULL)
            {
                fprintf(stderr, "No pcsr %zu-way partitioning found for: `%s`!\n", (size_t)pcount, base);
                goto free_dist_pcsr;
            }

            sor_pcsr(cb_benchmark_initialize_pcsr(o, target_graph))

            for (graph_size_t p = 0; cb_benchmark_initialize_par(o, p); p++)
                for (size_t d = 0; d < o->pcsr_fun_count; d++)
                {
                    soc(cb_benchmark_before_pcsr(o, target_graph, d))

                    for (size_t i = 0; i < o->start_iterations; i++)
                        sor_pcsr(cb_benchmark_disp_pcsr(o, target_graph, d))

                    if (!cb_benchmark_reset_time(o)) {
                        continue;
                    }

                    const time_mark_t start_time = time_mark();

                    for (size_t i = 0; i < o->bench_iterations; i++)
                        sor_pcsr(cb_benchmark_disp_pcsr(o, target_graph, d))

                    const time_diff_t duration = time_since(start_time) / o->bench_iterations;

                    sor_pcsr(cb_benchmark_after_pcsr(o, target_graph, d, duration))
                }

            free_pcsr:
            cb_benchmark_finalize_pcsr(o, target_graph);
            benchmark_pcsr_graph(_free)(target_graph);

            free_dist_pcsr:
            cb_benchmark_finalize_dist(o);
        }

    if (success && o->pcsc_fun_count > 0)
        for (graph_size_t pcount = o->bcount_begin; success && pcount <= o->bcount_end; pcount ARG_BENCH_NUM_BLOCKS_GROW)
        {
            soc(cb_benchmark_initialize_dist(o, pcount))

            benchmark_pcsc_graph_t *target_graph = benchmark_pcsc_graph(_read_file_transposed)(
                BENCH_PCSR_PART,
                pcount,
                BENCH_PCSR_BCOUNT,
                base,
                true,
                false,
                (graph_flags_enum_t) BENCH_GRAPH_FLAGS,
                NULL
            );

            if (target_graph == NULL)
            {
                fprintf(stderr, "No pcsc %zu-way partitioning found for: `%s`!\n", (size_t)pcount, base);
                goto free_dist_pcsc;
            }

            sor_pcsc(cb_benchmark_initialize_pcsc(o, target_graph))

            for (graph_size_t p = 0; cb_benchmark_initialize_par(o, p); p++)
                for (size_t d = 0; d < o->pcsc_fun_count; d++)
                {
                    soc(cb_benchmark_before_pcsc(o, target_graph, d))

                    for (size_t i = 0; i < o->start_iterations; i++)
                        sor_pcsc(cb_benchmark_disp_pcsc(o, target_graph, d))

                    if (!cb_benchmark_reset_time(o)) {
                        continue;
                    }
                    const time_mark_t start_time = time_mark();

                    for (size_t i = 0; i < o->bench_iterations; i++)
                        sor_pcsc(cb_benchmark_disp_pcsc(o, target_graph, d))

                    const time_diff_t duration = time_since(start_time) / o->bench_iterations;

                    sor_pcsc(cb_benchmark_after_pcsc(o, target_graph, d, duration))
                }

            free_pcsc:
            cb_benchmark_finalize_pcsc(o, target_graph);
            benchmark_pcsc_graph(_free)(target_graph);

            free_dist_pcsc:
            cb_benchmark_finalize_dist(o);
        }

    #undef soc
    #undef sor_pcsr
    #undef sor_pcsc

    return success;
}

static bool benchmark_run(benchmark_options_t *o)
{
    assert(o != NULL);

    if (o->eli_fun_count + o->csr_fun_count + o->csc_fun_count + o->bcsr_fun_count + o->bcsc_fun_count + o->pcsr_fun_count + o->pcsc_fun_count < 1)
        return false;

    bool success = true;

    for (graph_size_t vcount = o->vcount_begin; success && vcount && vcount <= o->vcount_end; vcount ARG_BENCH_NUM_VERTICES_GROW)
        for (graph_size_t ecount = o->ecount_begin; success && ecount && ecount <= o->ecount_end; ecount ARG_BENCH_NUM_EDGES_GROW)
        {
            if (!cb_benchmark_before_graph(o, "Erdos-Renyi")) {
                continue;
            }

            benchmark_csr_graph_t *graph = benchmark_csr_graph(_new_random)(vcount, vcount * ecount, (graph_flags_enum_t) BENCH_GRAPH_FLAGS);
            if (graph == NULL)
            {
                fprintf(stderr, "Cannot generate random graph [%zu, %zu]!\n", (size_t)vcount, (size_t)ecount);
                cb_benchmark_after_graph(o);
                continue;
            }

            success = benchmark_run_graph(o, graph);
            benchmark_csr_graph(_free)(graph);

            cb_benchmark_after_graph(o);
        }

    for (size_t file = 0; success && (o->files != NULL) && (file < o->files->we_wordc); file++)
    {
        if (!cb_benchmark_before_graph(o, o->files->we_wordv[file])) {
            continue;
        }

        if (success && o->eli_fun_count + o->csr_fun_count + o->csc_fun_count + o->bcsr_fun_count + o->bcsc_fun_count > 0)
        {
            benchmark_csr_graph_t *graph = benchmark_csr_graph(_read_file)(o->files->we_wordv[file], (graph_flags_enum_t) BENCH_GRAPH_FLAGS, NULL);

            if (graph == NULL)
                fprintf(stderr, "Invalid input file: `%s`!\n", o->files->we_wordv[file]);
            else
            {
                success = benchmark_run_graph(o, graph);
                benchmark_csr_graph(_free)(graph);
            }
        }

        if (success && o->pcsr_fun_count + success && o->pcsc_fun_count > 0)
            success = benchmark_run_graph_dist(o, o->files->we_wordv[file]);

        cb_benchmark_after_graph(o);
    }

    return success;
}
