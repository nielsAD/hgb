// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/pagerank.h"
#include "graph/pcsr_mpi.h"
#include "util/memory.h"
#include "util/mkl.h"
#include "util/mpi.h"
#include "util/string.h"
#include "util/starpu.h"
#include "util/openmp.h"
#include "util/cuda.h"

// benchmark.h graph definitions
#define benchmark_eli_graph_t    pr_eli_graph_t
#define benchmark_eli_graph(ID)  pr_eli_graph(ID)

#define benchmark_csr_graph_t    pr_csr_graph_t
#define benchmark_csc_graph_t    pr_csc_graph_t
#define benchmark_csr_graph(ID)  pr_csr_graph(ID)
#define benchmark_csc_graph(ID)  pr_csc_graph(ID)

#define benchmark_bcsr_graph_t   pr_bcsr_graph_t
#define benchmark_bcsc_graph_t   pr_bcsc_graph_t
#define benchmark_bcsr_graph(ID) pr_bcsr_graph(ID)
#define benchmark_bcsc_graph(ID) pr_bcsc_graph(ID)

#define benchmark_pcsr_graph_t   pr_pcsr_graph_t
#define benchmark_pcsc_graph_t   pr_pcsc_graph_t
#define benchmark_pcsr_graph(ID) pr_pcsr_graph(ID)
#define benchmark_pcsc_graph(ID) pr_pcsc_graph(ID)

static graph_size_t benchmark_comm_rank(void *o);
#define BENCH_GRAPH_FLAGS (E_GRAPH_FLAG_PIN | E_GRAPH_FLAG_DEG_IO)
#define BENCH_PCSR_PART benchmark_comm_rank(o)

#include "util/benchmark.h"

static const char *const argp_doc      = "Benchmark different pagerank implementations.";
static const char *const argp_args_doc = "[input_filename_1] ... [input_filename_N]";

const char *argp_program_version     = "ALPHA";
const char *argp_program_bug_address = "https://github.com/nielsAD";

enum PAGERANK_BENCHMARK_OPTIONS {
    OPT_LIST    = 'l',
    OPT_ENABLE  = 'y',
    OPT_DISABLE = 'n',

    OPT_DAMPING = 'd',
    OPT_EPSILON = 'p',
    OPT_STEPS   = 's',

    OPT_STEP_BEG = BENCH_OPT_USER + 1,
    OPT_STEP_END
};

static const struct argp_option argp_options[] = {
    // Inputs: generator and/or files
    ARG_BENCH_GENERATOR_INPUT,
    ARG_BENCH_FILE_INPUT,

    // Benchmark options: alg_iterations, blocks, etc.
    ARG_BENCH_OPTIONS,

    // Function options
    { "list", OPT_LIST, NULL, OPTION_ARG_OPTIONAL, "List the names of the pagerank implementations.", 0},
    { "enable_fun",  OPT_ENABLE,  "pattern", 0, "Enable benchmark function(s) with glob pattern. Use '--list' to see possible options.", 0},
    { "disable_fun", OPT_DISABLE, "pattern", 0, "Disable benchmark function(s) with glob pattern. Use '--list' to see possible options.", 0},

    // Pagerank options
    { "damping", OPT_DAMPING, "float", 0, "Pagerank damping parameter.", 0},
    { "epsilon", OPT_EPSILON, "float", 0, "Pagerank epsilon parameter.", 0},
    { "steps",   OPT_STEPS,   "int",   0, "Number of pagerank supersteps (0 for convergence).", 0},

    {"num_steps_begin", OPT_STEP_BEG, "int", 0, "Number of supersteps begin loop.", 0},
    {"sb",              OPT_STEP_BEG, "int", OPTION_ALIAS, NULL, 0},
    {"num_steps_end",   OPT_STEP_END, "int", 0, "Number of supersteps end loop.", 0},
    {"se",              OPT_STEP_END, "int", OPTION_ALIAS, NULL, 0},

    { NULL, 0, NULL, 0, NULL, 0}
};

#ifndef ARG_NUM_STEPS_GROW
   #define ARG_NUM_STEPS_GROW *= 2
#endif

typedef struct pagerank_benchmark_options {
    // Extend the benchmark options struct
    benchmark_options_t benchmark_options;

    // Pagerank benchmark options
    uint32_t steps_begin;
    uint32_t steps_end;

    // Pagerank execution options
    pagerank_options_t pagerank_options;

    // Benchmark vars
    MPI_Comm comm;
    const char *gid;
    pr_float *ref;
    graph_size_t size;
    uint32_t alg_iterations;
    bool check_result;
} pagerank_benchmark_options_t;

static error_t argp_parser(int key, char *arg, struct argp_state *state) {
    pagerank_benchmark_options_t *o = (pagerank_benchmark_options_t*) state->input;

    static bool force_random = false;
    static bool first_enable = true;

    switch (key)
    {
        case OPT_LIST:
        {
            if (mpi_is_root())
                print_pagerank_solvers();
            exit(EXIT_SUCCESS);
        }

        case OPT_ENABLE:
        case OPT_DISABLE:
        {
            if (first_enable)
            {
                // Toggle all solvers to true (disable pattern) or false (enable pattern)
                toggle_pagerank_solvers(key == OPT_DISABLE);
                first_enable = false;
            }

            if (arg != NULL && arg[0] == '!')
                toggle_pagerank_solvers_by_pattern(arg + 1, /* match = */ false, key == OPT_ENABLE);
            else
                toggle_pagerank_solvers_by_pattern(arg + 0, /* match = */ true, key == OPT_ENABLE);

            break;
        }

        case OPT_DAMPING: o->pagerank_options.damping = strtold(arg, NULL); break;
        case OPT_EPSILON: o->pagerank_options.epsilon = strtold(arg, NULL); break;

        case OPT_STEPS:    o->steps_begin = o->steps_end = strtoullr(arg, o->steps_begin, o->steps_end); break;
        case OPT_STEP_BEG: o->steps_begin =                strtoullr(arg, o->steps_begin, o->steps_end); break;
        case OPT_STEP_END:                  o->steps_end = strtoullr(arg, o->steps_begin, o->steps_end); break;

        // Treat key-less parameters as file inputs, disable random generator
        case ARGP_KEY_ARG:
        {
            if (!force_random)
                benchmark_argp_parser(BENCH_OPT_GENERATOR, (char*)"false", state);
            return benchmark_argp_parser(BENCH_OPT_FILE_INPUT, arg, state);
        }

        // Following keys fall through to default behavior
        case BENCH_OPT_GENERATOR:
            if (strtob(arg, true) == true)
                force_random = true;

            fallthrough;
        default:
            return benchmark_argp_parser(key, arg, state);
    }

    return 0;
}

static graph_size_t benchmark_comm_rank(void *o)
{
    assert(o != NULL);
    MPI_Comm comm = ((pagerank_benchmark_options_t*)o)->comm;
    if (comm == MPI_COMM_NULL)
        return 0;

    int r;
    MPI_Comm_rank(comm, &r);
    return (graph_size_t) r;
}

static bool initialize_dst(benchmark_options_t *o, const graph_size_t vc, const graph_size_t oh)
{
    assert(o != NULL);

    MPI_Barrier(MPI_COMM_WORLD);
    return_if_not_mpi_root(true);

    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    po->pagerank_options.result = memory_pinned_talloc_if(
        E_GRAPH_FLAG_PIN & BENCH_GRAPH_FLAGS,
        pr_float,
        vc + oh
    );
    po->size = vc;

    return vc > 0;
}

bool assert_equal_to_ref(pagerank_benchmark_options_t *po, const graph_size_t vc)
{
    unsigned int dif = 0;
    long double  sum = 0.0;
    for (graph_size_t v = 0; v < vc; v++)
    {
        if (pr_abs(po->ref[v] - po->pagerank_options.result[v]) > (PR_EPSILON * 10))
            dif++;
        sum += po->pagerank_options.result[v];
    }

    long double difp = (long double)(dif) / vc;
    bool       equal = (difp <= 0.05) && (fabsl(sum - 1.0) <= 1e-3);

    if (!equal)
    {
        mpi_printf("# NO MATCH (diff=%.0Lf%%, sum=%Lf)", difp*100, sum);

        #ifdef NDEBUG
            equal = po->steps_begin > 0 && po->steps_begin <= po->steps_end;
        #else
            fprintf(stderr, "\nERROR: RESULT NOT EQUAL TO REFERENCE (diff=%Lf >= %Lf, sum=|%Lf-1.0| >= %Lf)\n",
                (long double) difp,
                (long double) 0.05,
                (long double) sum,
                (long double) 1e-3
            );
            for (graph_size_t v = 0; v < MIN(vc, (graph_size_t)10); v++)
                fprintf(stderr, "[%zu] %12.10Lf <> %12.10Lf\n",
                    (size_t) v,
                    (long double) po->ref[v],
                    (long double) po->pagerank_options.result[v]
                );
        #endif
    }

    return equal;
}

static bool assert_and_finalize_dst(benchmark_options_t *o, const double d, bool check_result)
{
    assert(o != NULL);

    for (unsigned n = 0; n < starpu_memory_nodes_get_count(); n++)
        _starpu_free_all_automatically_allocated_buffers(n);

    return_if_not_mpi_root(true);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    const uint32_t ben_it = po->benchmark_options.bench_iterations;
    const uint32_t alg_it = po->alg_iterations;
    mpi_printf("\t%3u\t%10.3Lf\t%10.3Lf\t%10.3Lf\t%10.3Lf\t%10.3Lf\t%10.3Lf",
        alg_it / ben_it,
        time_ms(d),
        time_ms(po->pagerank_options.stage_time[E_PR_STAGE_INIT]) / ben_it,
        time_us(po->pagerank_options.stage_time[E_PR_STAGE_BASERANK]) / alg_it,
        time_us(po->pagerank_options.stage_time[E_PR_STAGE_UPDATE]) / alg_it,
        time_us(po->pagerank_options.stage_time[E_PR_STAGE_DIFF]) / alg_it,
        time_ms(po->pagerank_options.stage_time[E_PR_STAGE_TRANSFER]) / ben_it
    );

    if (check_result && po->ref == NULL)
        po->ref = po->pagerank_options.result;

    const bool equal_to_ref = !check_result || po->alg_iterations == 0 || assert_equal_to_ref(po, po->size);

    if (po->pagerank_options.result != po->ref)
        memory_pinned_free_if(E_GRAPH_FLAG_PIN & BENCH_GRAPH_FLAGS, po->pagerank_options.result);

    mpi_printf("\n");
    return equal_to_ref;
}

static bool cb_benchmark_initialize_eli (UNUSED benchmark_options_t *o, benchmark_eli_graph_t *g) { assert(g != NULL); return g->ecount > 0; }
static bool cb_benchmark_initialize_csr (UNUSED benchmark_options_t *o, benchmark_csr_graph_t *g) { assert(g != NULL); return g->ecount > 0; }
static bool cb_benchmark_initialize_csc (UNUSED benchmark_options_t *o, benchmark_csc_graph_t *g) { assert(g != NULL); return g->ecount > 0; }

static void cb_benchmark_finalize_eli (UNUSED benchmark_options_t *o, UNUSED benchmark_eli_graph_t *g)  { /* do nothing */ }
static void cb_benchmark_finalize_csr (UNUSED benchmark_options_t *o, UNUSED benchmark_csc_graph_t *g)  { /* do nothing */ }
static void cb_benchmark_finalize_bcsr(UNUSED benchmark_options_t *o, UNUSED benchmark_bcsr_graph_t *g) { /* do nothing */ }
static void cb_benchmark_finalize_bcsc(UNUSED benchmark_options_t *o, UNUSED benchmark_bcsc_graph_t *g) { /* do nothing */ }
static void cb_benchmark_finalize_pcsr(UNUSED benchmark_options_t *o, UNUSED benchmark_pcsr_graph_t *g) { /* do nothing */ }
static void cb_benchmark_finalize_pcsc(UNUSED benchmark_options_t *o, UNUSED benchmark_pcsc_graph_t *g) { /* do nothing */ }

static bool cb_benchmark_before_graph(benchmark_options_t *o, const char *const n)
{
    assert(o != NULL);

    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;
    po->comm  = MPI_COMM_NULL;
    po->gid  = basename((char*)n);
    po->ref  = NULL;
    po->size = 0;

    po->pagerank_options.result = NULL;

    return true;
}

static bool cb_benchmark_reset_time(benchmark_options_t *o) {
    assert(o != NULL);

    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    po->alg_iterations = 0;

    for (pagerank_stage_t i = 0; i < E_PR_STAGE_MAX; i++)
        po->pagerank_options.stage_time[i] = 0;

    return true;
}

static void cb_benchmark_after_graph(benchmark_options_t *o)
{
    assert(o != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    if (po->ref != NULL)
        memory_pinned_free_if(E_GRAPH_FLAG_PIN & BENCH_GRAPH_FLAGS, po->ref);
}

static bool cb_benchmark_initialize_par(benchmark_options_t *o, const graph_size_t p)
{
    assert(o != NULL);

    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;
    if (po->steps_begin > po->steps_end || po->steps_begin == 0)
        return p == 0;

    if (p == 0)
    {
        po->pagerank_options.min_iterations = po->pagerank_options.max_iterations = po->steps_begin;
        return true;
    }
    else
    {
        po->pagerank_options.min_iterations = po->pagerank_options.max_iterations ARG_NUM_STEPS_GROW;
        return po->pagerank_options.min_iterations <= po->steps_end;
    }

    return false;
}

static bool cb_benchmark_initialize_dist(benchmark_options_t *o, const graph_size_t p)
{
    assert(o != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    po->comm = mpi_get_sub_comm(MPI_COMM_WORLD, p);
    return (graph_size_t)mpi_get_size() >= p;
}

static void cb_benchmark_finalize_dist(benchmark_options_t *o)
{
    assert(o != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    if (po->comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&po->comm);
        po->comm = MPI_COMM_NULL;
    }
}

static bool cb_benchmark_initialize_bcsr(UNUSED benchmark_options_t *o, benchmark_bcsr_graph_t *g)
{
    assert(g != NULL);

    bcsr_forall_blocks(b, g)
    {
        if (g->blocks[b]->vcount < 1)
        {
            mpi_fprintf(stderr, "ERROR: EMPTY BLOCK %zu IN %zu-WAY PARTITIONING.\n", (size_t)b, (size_t)g->bcount);
            return false;
        }
    }

    return g->ecount > 0;
}

static bool cb_benchmark_initialize_pcsr(benchmark_options_t *o, benchmark_pcsr_graph_t *g)
{
    assert(o != NULL);
    assert(g != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    if (po->comm == MPI_COMM_NULL)
        return true;

    pr_pcsr_graph(_filter_cross_graph)(g);
    if (g->local_graph->flags & E_GRAPH_FLAG_DEG_I)
        pr_pcsr_graph(_sync_degrees_in)(g, po->comm);
    if (g->local_graph->flags & E_GRAPH_FLAG_DEG_O)
        pr_pcsr_graph(_sync_degrees_out)(g, po->comm);

    g->vcount = benchmark_pcsr_graph(_get_global_vcount)(g, po->comm, /*with_ghost*/ false);
    return g->vcount > 0;
}

static bool cb_benchmark_initialize_bcsc (UNUSED benchmark_options_t *o, benchmark_bcsc_graph_t *g) { return cb_benchmark_initialize_bcsr(o, g); }
static bool cb_benchmark_initialize_pcsc (UNUSED benchmark_options_t *o, benchmark_pcsc_graph_t *g) { return cb_benchmark_initialize_pcsr(o, g); }

static void cb_benchmark_finalize_csc(UNUSED benchmark_options_t *o, benchmark_csr_graph_t *g)
{
    assert(g != NULL);
    if (memory_get_default_pinned_manager() & (E_MM_CUDA | E_MM_OPENCL))
        benchmark_csr_graph(_toggle_flag)(g, E_GRAPH_FLAG_PIN, false);
}

static bool cb_benchmark_before_eli(benchmark_options_t *o, const benchmark_eli_graph_t *g, UNUSED const size_t fun)
{
    assert(o != NULL);
    assert(g != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    if (po->size == 0)
    {
        benchmark_eli_graph(_get_vertex_range)(g, NULL, &po->size);
        po->size += 1;
    }

    return initialize_dst(o, po->size, 64);
}

static bool cb_benchmark_before_csr(benchmark_options_t *o, const benchmark_csr_graph_t *g, UNUSED const size_t fun)
{
    assert(g != NULL);
    return initialize_dst(o, g->vcount, 64);
}

static bool cb_benchmark_before_csc(benchmark_options_t *o, const benchmark_csc_graph_t *g, UNUSED const size_t fun)
{
    assert(g != NULL);
    return initialize_dst(o, g->vcount, 64);
}

static bool cb_benchmark_before_bcsr(benchmark_options_t *o, const benchmark_bcsr_graph_t *g, UNUSED const size_t fun)
{
    assert(g != NULL);
    return initialize_dst(o, g->vcount, 64 + g->bcount*BCSR_GRAPH_VERTEX_PACK);
}

static bool cb_benchmark_before_bcsc(benchmark_options_t *o, const benchmark_bcsc_graph_t *g, UNUSED const size_t fun)
{
    assert(g != NULL);
    return initialize_dst(o, g->vcount, 64 + g->bcount*BCSR_GRAPH_VERTEX_PACK);
}

static bool cb_benchmark_before_pcsr(benchmark_options_t *o, const benchmark_pcsr_graph_t *g, UNUSED const size_t fun)
{
    assert(g != NULL);
    return initialize_dst(o, g->vcount, 64);
}

static bool cb_benchmark_before_pcsc(benchmark_options_t *o, const benchmark_pcsc_graph_t *g, UNUSED const size_t fun)
{
    assert(g != NULL);
    return initialize_dst(o, g->vcount, 64);
}

static bool cb_benchmark_disp_eli(benchmark_options_t *o, const benchmark_eli_graph_t *g, const size_t fun)
{
    assert(o != NULL);
    assert(g != NULL);

    if (!eli_pagerank_solvers[fun].active)
        return true;

    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    memset(po->pagerank_options.result, 0, sizeof(*po->pagerank_options.result) * po->size);
    po->alg_iterations += eli_pagerank_solvers[fun].func(g, &po->pagerank_options);

    return true;
}

static bool cb_benchmark_disp_csr(benchmark_options_t *o, const benchmark_csr_graph_t *g, const size_t fun)
{
    assert(o != NULL);
    assert(g != NULL);

    if (!csr_pagerank_solvers[fun].active)
        return true;

    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    memset(po->pagerank_options.result, 0, sizeof(*po->pagerank_options.result) * po->size);
    po->alg_iterations += csr_pagerank_solvers[fun].func(g, &po->pagerank_options);

    return true;
}

static bool cb_benchmark_disp_csc(benchmark_options_t *o, const benchmark_csc_graph_t *g, const size_t fun)
{
    assert(o != NULL);
    assert(g != NULL);

    if (!csc_pagerank_solvers[fun].active)
        return true;

    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    memset(po->pagerank_options.result, 0, sizeof(*po->pagerank_options.result) * po->size);
    po->alg_iterations += csc_pagerank_solvers[fun].func(g, &po->pagerank_options);

    return true;
}

static bool cb_benchmark_disp_bcsr(benchmark_options_t *o, const benchmark_bcsr_graph_t *g, const size_t fun)
{
    assert(o != NULL);
    assert(g != NULL);

    if (!bcsr_pagerank_solvers[fun].active)
        return true;

    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    memset(po->pagerank_options.result, 0, sizeof(*po->pagerank_options.result) * po->size);
    po->alg_iterations += bcsr_pagerank_solvers[fun].func(g, &po->pagerank_options);

    return true;
}

static bool cb_benchmark_disp_bcsc(benchmark_options_t *o, const benchmark_bcsc_graph_t *g, const size_t fun)
{
    assert(o != NULL);
    assert(g != NULL);

    if (!bcsc_pagerank_solvers[fun].active)
        return true;

    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    memset(po->pagerank_options.result, 0, sizeof(*po->pagerank_options.result) * po->size);
    po->alg_iterations += bcsc_pagerank_solvers[fun].func(g, &po->pagerank_options);

    return true;
}

static bool cb_benchmark_disp_pcsr(benchmark_options_t *o, const benchmark_pcsr_graph_t *g, const size_t fun)
{
    assert(o != NULL);
    assert(g != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    if (!pcsr_pagerank_solvers[fun].active || po->comm == MPI_COMM_NULL)
        return true;

    memset(po->pagerank_options.result, 0, sizeof(*po->pagerank_options.result) * po->size);
    po->alg_iterations += pcsr_pagerank_solvers[fun].func(po->comm, g, &po->pagerank_options);

    return true;
}

static bool cb_benchmark_disp_pcsc(benchmark_options_t *o, const benchmark_pcsc_graph_t *g, const size_t fun)
{
    assert(o != NULL);
    assert(g != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    if (!pcsc_pagerank_solvers[fun].active || po->comm == MPI_COMM_NULL)
        return true;

    memset(po->pagerank_options.result, 0, sizeof(*po->pagerank_options.result) * po->size);
    po->alg_iterations += pcsc_pagerank_solvers[fun].func(po->comm, g, &po->pagerank_options);

    return true;
}

static bool cb_benchmark_after_eli(benchmark_options_t *o, const benchmark_eli_graph_t *g, const size_t fun, const double duration)
{
    assert(o != NULL);
    assert(g != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    mpi_printf("%-30s\t%25s\t%3d\t%9d\t%9zu", po->gid, eli_pagerank_solvers[fun].name, 1, -1, (size_t)g->ecount);
    return assert_and_finalize_dst(o, duration, eli_pagerank_solvers[fun].check_result);
}

static bool cb_benchmark_after_csr(benchmark_options_t *o, const benchmark_csr_graph_t *g, const size_t fun, const double duration)
{
    assert(o != NULL);
    assert(g != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    mpi_printf("%-30s\t%25s\t%3d\t%9zu\t%9zu", po->gid, csr_pagerank_solvers[fun].name, 1, (size_t)g->vcount, (size_t)g->ecount);
    return assert_and_finalize_dst(o, duration, csr_pagerank_solvers[fun].check_result);
}

static bool cb_benchmark_after_csc(benchmark_options_t *o, const benchmark_csc_graph_t *g, const size_t fun, const double duration)
{
    assert(o != NULL);
    assert(g != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    mpi_printf("%-30s\t%25s\t%3d\t%9zu\t%9zu", po->gid, csc_pagerank_solvers[fun].name, 1, (size_t)g->vcount, (size_t)g->ecount);
    return assert_and_finalize_dst(o, duration, csc_pagerank_solvers[fun].check_result);
}

static bool cb_benchmark_after_bcsr(benchmark_options_t *o, const benchmark_bcsr_graph_t *g, const size_t fun, const double duration)
{
    assert(o != NULL);
    assert(g != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    mpi_printf("%-30s\t%25s\t%3zu\t%9zu\t%9zu", po->gid, bcsr_pagerank_solvers[fun].name, (size_t)g->bcount, (size_t)g->vcount, (size_t)g->ecount);
    return assert_and_finalize_dst(o, duration, bcsr_pagerank_solvers[fun].check_result);
}

static bool cb_benchmark_after_bcsc(benchmark_options_t *o, const benchmark_bcsc_graph_t *g, const size_t fun, const double duration)
{
    assert(o != NULL);
    assert(g != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    mpi_printf("%-30s\t%25s\t%3zu\t%9zu\t%9zu", po->gid, bcsc_pagerank_solvers[fun].name, (size_t)g->bcount, (size_t)g->vcount, (size_t)g->ecount);
    return assert_and_finalize_dst(o, duration, bcsc_pagerank_solvers[fun].check_result);
}

static bool cb_benchmark_after_pcsr(benchmark_options_t *o, const benchmark_pcsr_graph_t *g, const size_t fun, const double duration)
{
    assert(o != NULL);
    assert(g != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    graph_size_t ecount = (po->comm == MPI_COMM_NULL)
        ? 0
        : benchmark_pcsr_graph(_get_global_ecount)(g, po->comm, /*with_cross*/ true);

    mpi_printf("%-30s\t%25s\t%3zu\t%9zu\t%9zu", po->gid, pcsr_pagerank_solvers[fun].name, (size_t)g->pcount, (size_t)g->vcount, (size_t)ecount);
    return assert_and_finalize_dst(o, duration, pcsr_pagerank_solvers[fun].check_result);
}

static bool cb_benchmark_after_pcsc(benchmark_options_t *o, const benchmark_pcsc_graph_t *g, const size_t fun, const double duration)
{
    assert(o != NULL);
    assert(g != NULL);
    pagerank_benchmark_options_t *po = (pagerank_benchmark_options_t*) o;

    graph_size_t ecount = (po->comm == MPI_COMM_NULL)
        ? 0
        : benchmark_pcsc_graph(_get_global_ecount)(g, po->comm, /*with_cross*/ true);

    mpi_printf("%-30s\t%25s\t%3zu\t%9zu\t%9zu", po->gid, pcsc_pagerank_solvers[fun].name, (size_t)g->pcount, (size_t)g->vcount, (size_t)ecount);
    return assert_and_finalize_dst(o, duration, pcsc_pagerank_solvers[fun].check_result);
}

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);

    mkl_initialize();
    mpi_initialize(&argc, &argv);
    starpu_inititialize_conf(argc, argv);

    static pagerank_benchmark_options_t options;
    options.pagerank_options.damping = 0.85;
    options.pagerank_options.epsilon = 1e-6;

    options.pagerank_options.local_iterations = 2;
    options.pagerank_options.min_iterations   = 1;
    options.pagerank_options.max_iterations   = 250;

    options.benchmark_options.start_iterations = 1;
    options.benchmark_options.bench_iterations = 5;

    options.benchmark_options.bcount_begin = 1;        // 2 ** 0
    options.benchmark_options.bcount_end   = 16;       // 2 ** 4
    options.benchmark_options.vcount_begin = 262144;   // 2 ** 18
    options.benchmark_options.vcount_end   = 16777216; // 2 ** 24
    options.benchmark_options.ecount_begin = 1;        // 2 ** 0
    options.benchmark_options.ecount_end   = 256;      // 2 ** 8

    const struct argp argp = {argp_options, argp_parser, argp_args_doc, argp_doc, NULL, NULL, NULL};
    if (argp_parse(&argp, argc, argv, 0, NULL, &options) != 0)
    {
        mpi_printf("Could not parse application arguments.\n");
        return EXIT_FAILURE;
    }

    pagerank_initialize();
    reorder_active_pagerank_solvers();

    options.benchmark_options.eli_fun_count = count_active_eli_pagerank_solvers();
    options.benchmark_options.csr_fun_count = count_active_csr_pagerank_solvers();
    options.benchmark_options.csc_fun_count = count_active_csc_pagerank_solvers();
    options.benchmark_options.bcsr_fun_count = count_active_bcsr_pagerank_solvers();
    options.benchmark_options.bcsc_fun_count = count_active_bcsc_pagerank_solvers();
    options.benchmark_options.pcsr_fun_count = count_active_pcsr_pagerank_solvers();
    options.benchmark_options.pcsc_fun_count = count_active_pcsc_pagerank_solvers();

    if (!mpi_is_root())
        toggle_pagerank_solvers_by_pattern("*mpi*", /* match = */ false, false);

    fprintf(stderr, "damping    = %4Lg  epsilon    = %Lg\n", (long double) options.pagerank_options.damping, (long double) options.pagerank_options.epsilon);
    fprintf(stderr, "iterations = %4zu  warming_up = %4zu\n", options.benchmark_options.bench_iterations, options.benchmark_options.start_iterations);
    fprintf(stderr, "|CSR|= %zu |CSC|= %zu |BCSR|= %zu |BCSC|= %zu |PCSR|= %zu |PCSC|= %zu |ELI|= %zu\n",
        options.benchmark_options.csr_fun_count,  options.benchmark_options.csc_fun_count,
        options.benchmark_options.bcsr_fun_count, options.benchmark_options.bcsc_fun_count,
        options.benchmark_options.pcsr_fun_count, options.benchmark_options.pcsc_fun_count,
        options.benchmark_options.eli_fun_count);
    fprintf(stderr, "|F| = %zu\n", (options.benchmark_options.files == NULL) ? 0 : options.benchmark_options.files->we_wordc);
    fprintf(stderr, "|B| = [%8zu, %8zu] -> %s\n", (size_t)options.benchmark_options.bcount_begin, (size_t)options.benchmark_options.bcount_end, STATIC_STR(ARG_BENCH_NUM_BLOCKS_GROW));
    fprintf(stderr, "|V| = [%8zu, %8zu] -> %s\n", (size_t)options.benchmark_options.vcount_begin, (size_t)options.benchmark_options.vcount_end, STATIC_STR(ARG_BENCH_NUM_VERTICES_GROW));
    fprintf(stderr, "|E| = [%8zu, %8zu] -> %s\n", (size_t)options.benchmark_options.ecount_begin, (size_t)options.benchmark_options.ecount_end, STATIC_STR(ARG_BENCH_NUM_EDGES_GROW));
    fprintf(stderr, "|S| = [%8u, %8u] -> %s\n",   options.steps_begin, options.steps_end, STATIC_STR(ARG_NUM_STEPS_GROW));

    fprintf(stderr, "[%d] CPU/CUDA/OCL/OMP/MPI = %u %u %u %d %d\n", mpi_get_rank() + 1, starpu_cpu_worker_get_count(), starpu_cuda_worker_get_count(), starpu_opencl_worker_get_count(), omp_get_max_threads(), mpi_get_size());
    starpu_topology_print(stderr);

    if (starpu_cuda_worker_get_count() > 0)
        cuda_initialize();

    MPI_Barrier(MPI_COMM_WORLD);

    mpi_printf("%-30s\t%25s\t%3s\t%9s\t%9s\t%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n", "DATASET", "SOLVER", "|B|", "|V|", "|E|", "it", "time_ms", "init_ms", "base_us", "rank_us", "diff_us", "trans_ms");
    const bool res = benchmark_run(&options.benchmark_options);

    if (starpu_cuda_worker_get_count() > 0)
        cuda_initialize();

    pagerank_finalize();
    starpu_finalize();
    mpi_finalize();
    mkl_finalize();

    if (options.benchmark_options.files != NULL)
    {
        wordfree(options.benchmark_options.files);
        free(options.benchmark_options.files);
    }

    return res ? EXIT_SUCCESS : EXIT_FAILURE;
}
