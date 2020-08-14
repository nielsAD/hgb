// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/pcsr.h"
#include "util/memory.h"
#include "util/file.h"
#include "util/math.h"
#include "util/string.h"
#include "util/openmp.h"

#include <argp.h>
#include <unordered_map>

#define stride_graph_t   csr_graph_v_t
#define stride_graph(ID) csr_graph_v ## ID

static const char *const argp_doc      = "Output stride info for input graph";
static const char *const argp_args_doc = "[input_filename]";

const char *argp_program_version     = "ALPHA";
const char *argp_program_bug_address = "https://github.com/nielsAD";

enum STRIDE_OPTIONS {
    OPT_TRANSPOSE   = 't',

    OPT_MMAP        = 'D',
    OPT_FORMAT      = 'f',
    OPT_INPUT_FMT   = 'I',
    OPT_INPUT_FILE  = 'i',
};

static const struct argp_option argp_options[] = {
    { "transpose",    OPT_TRANSPOSE,  "bool", OPTION_ARG_OPTIONAL, "Transpose input graph.", 0},

    { "input_format", OPT_INPUT_FMT,  "string", 0, "Force reader to assume this graph format (e.g. `dimacs`,`csr`,`el`).", 0},
    { "format",       OPT_FORMAT,     "string", OPTION_ALIAS, NULL, 0},
 
    { "input",        OPT_INPUT_FILE, "filename", 0, "Input filename.", 0},

    { "disk",   OPT_MMAP, "directory", OPTION_ARG_OPTIONAL, "Specify directory to store intermediates in temporary files to reduce memory load. Working dir by default.", 0},

    {NULL, 0, NULL, 0, NULL, 0}
};

typedef struct stride_options {
    // Options
    bool transpose;

    // Input / output handling
    char *input_file;
    char *input_ext;
} stride_options_t;

static error_t argp_parser(int key, char *arg, struct argp_state *state) {
    stride_options_t *o = (stride_options_t*) state->input;

    switch (key)
    {
        case OPT_TRANSPOSE: o->transpose = strtob(arg, true); break;

        case OPT_MMAP:
        {
            memory_set_default_pinned_manager(E_MM_MMAP);
            memory_set_mapped_tmpdir(arg);
            break;
        }

        case OPT_FORMAT:
        case OPT_INPUT_FMT:
        {
            if (o->input_ext == NULL)
                o->input_ext = strdup(arg);
            else
                return ARGP_ERR_UNKNOWN;
            break;
        }

        case OPT_INPUT_FILE:
        case ARGP_KEY_ARG:
        {
            if (o->input_file == NULL)
                o->input_file = strdup(arg);
            else
                return ARGP_ERR_UNKNOWN;
            break;
        }

        default:
            return ARGP_ERR_UNKNOWN;
    }

    return 0;
}

int calculate_stride(const stride_graph_t *graph)
{
    uint64_t stride_cnt  = 0;
    uint64_t stride_vert = 0;
    uint64_t stride_glob = 0;
    uint64_t stride_diag = 0;

    std::unordered_map<graph_size_t, std::pair<graph_size_t, size_t>> cache;
    graph_size_t cache_time  = 0;
    graph_size_t cache_local = 0;

    graph_size_t last_fr = 0;
    graph_size_t last_to = 0;
    csr_forall_edges(eid,efr,eto,graph)
    {
        const graph_size_t lane = eto / 16;
        const size_t       mask = 2 << (eto % 16);

        auto idx = cache.find(lane);
        if ((idx == cache.end()) || ((eid - idx->second.first) > 4096))
        {
            cache[lane] = {eid, eto % 16};
        }
        else if (idx->second.second & mask)
        {
            cache_time++;
            idx->second.first = eid;
        }
        else {
            cache_local++;
            idx->second.first   = eid;
            idx->second.second |= mask;
        }

        if (efr == last_fr)
        {
            stride_vert += (uint64_t) labs((long int)last_to - (long int)eto);
            stride_cnt++;
        }

        stride_glob += (uint64_t) labs((long int)last_to - (long int)eto);
        stride_diag += (uint64_t) labs((long int)efr  - (long int)eto);
        last_fr = efr;
        last_to = eto;
    }


    stride_vert /= (uint64_t)stride_cnt;
    stride_glob /= (uint64_t)graph->ecount;
    stride_diag /= (uint64_t)graph->ecount;

    uint64_t pack = ((uint64_t)graph->vcount * graph->vcount) / (uint64_t)graph->ecount;
    int64_t pack_vert = (int64_t)((stride_vert * 100) / pack);
    int64_t pack_glob = (int64_t)((stride_glob * 100) / pack);
    int64_t pack_diag = (int64_t)((stride_diag * 100) / pack);

    double hit_time  = (cache_time / (double)graph->ecount) * 100.0;
    double hit_local = (cache_local / (double)graph->ecount) * 100.0;

    printf("Packing: %lu\n", pack);
    printf("Cache hit: %zu (time) / %zu (local) / %zu (miss)\n", (size_t) cache_time, (size_t) cache_local, (size_t) (graph->ecount - cache_time - cache_local));
    printf("Cache hit: %.1f%% (time) / %.1f%% (local) = %.f%%\n", hit_time, hit_local, hit_time+hit_local);
    printf("Vertex stride: %lu (%ld%%)\n", stride_vert, pack_vert);
    printf("Global stride: %lu (%ld%%)\n", stride_glob, pack_glob);
    printf("Diagonal stride: %lu (%ld%%)\n", stride_diag, pack_diag);

    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    static stride_options_t options;

    const struct argp argp = {argp_options, argp_parser, argp_args_doc, argp_doc, NULL, NULL, NULL};
    if (argp_parse(&argp, argc, argv, 0, NULL, &options) != 0)
    {
        fprintf(stderr, "Could not parse application arguments.\n");
        return EXIT_FAILURE;
    }

    int res = EXIT_SUCCESS;

    stride_graph_t *graph = stride_graph(_read_file)(
        options.input_file,
        (graph_flags_enum_t) (E_GRAPH_FLAG_PIN),
        options.input_file || options.input_ext ? options.input_ext : "el"
    );

    if (options.transpose)
        stride_graph(_transpose)(graph);

    if (graph == NULL || graph->vcount < 1 || graph->ecount < 1)
    {
        fprintf(stderr, "Invalid input file `%s`!\n", options.input_file);
        res = EXIT_FAILURE;
    }
    else
    {
        fprintf(stderr, "G(|V|, |E|) = G(%zu, %zu)\n", (size_t)graph->vcount, (size_t)graph->ecount);
        
        printf("Degree assortativity: %f\n", stride_graph(_degree_assortativity)(graph));
        printf("Clustering coefficient: %f\n", stride_graph(_avg_clustering_coefficient)(graph));
        printf("Avg neighbor degree: %f (in) / %f (out)\n", stride_graph(_avg_neighbor_degree_in)(graph), stride_graph(_avg_neighbor_degree_out)(graph));

        res = calculate_stride(graph);
        stride_graph(_free)(graph);
    }

    free(options.input_file);
    free(options.input_ext);

    return res;
}
