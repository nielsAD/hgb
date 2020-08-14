// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/csr.h"
#include "util/memory.h"
#include "util/file.h"
#include "util/string.h"

#include <argp.h>

#define cleaner_graph_t   csr_graph_v_t
#define cleaner_graph(ID) csr_graph_v ## ID

static const char *const argp_doc      = "Sanitizes graph structure and converts between graph formats.";
static const char *const argp_args_doc = "[input_filename] [output_filename]";

const char *argp_program_version     = "ALPHA";
const char *argp_program_bug_address = "https://github.com/nielsAD";

enum CLEANER_OPTIONS {
    OPT_ALIGN_EDGES   = 'a',
    OPT_TO_DIRECTED   = 'd',
    OPT_TO_UNDIRECTED = 'u',
    OPT_SORT          = 's',

    OPT_TRANSPOSE = 't',
    OPT_UNIQUE    = 'q',
    OPT_NO_LOOPS  = 'l',
    OPT_CONNECTED = 'c',

    OPT_DIMACS = 'm',
    OPT_MMAP   = 'D',

    OPT_FORMAT      = 'f',
    OPT_INPUT_FMT   = 'I',
    OPT_OUTPUT_FMT  = 'O',

    OPT_INPUT_FILE  = 'i',
    OPT_OUTPUT_FILE = 'o'
};

static const struct argp_option argp_options[] = {
    { "align",         OPT_ALIGN_EDGES,  "int", 0, "Align the number of edges for every vertex to a multipe of this number.", 0},
    { "to_directed",   OPT_TO_DIRECTED,   NULL, OPTION_ARG_OPTIONAL, "For every edge (x:y), also add (y:x) to the graph.", 0},
    { "to_undirected", OPT_TO_UNDIRECTED, NULL, OPTION_ARG_OPTIONAL, "Only keep edges (x:y) where (x <= y).", 0},
    { "sort",          OPT_SORT,          NULL, OPTION_ARG_OPTIONAL, "Sort vertices by out-degree.", 0},

    { "transpose", OPT_TRANSPOSE, NULL, OPTION_ARG_OPTIONAL, "Reverse the direction of all edges.", 0},
    { "unique",    OPT_UNIQUE,    NULL, OPTION_ARG_OPTIONAL, "Remove duplicate edges.", 0},
    { "noloops",   OPT_NO_LOOPS,  NULL, OPTION_ARG_OPTIONAL, "Remove self loops.", 0},
    { "connected", OPT_CONNECTED, NULL, OPTION_ARG_OPTIONAL, "Remove unconnected vertices.", 0},

    { "dimacs", OPT_DIMACS, NULL, OPTION_ARG_OPTIONAL, "Set settings according to the DIMACS specification (produces valid input for e.g. Metis, Chaco, KaHIP).", 0},

    { "format",        OPT_FORMAT,     "string",    0, "Force reader/writer to use this format (e.g. `dimacs`,`csr`,`el`).", 0},
    { "input_format",  OPT_INPUT_FMT,  "string",    0, "Force reader to assume this graph format (e.g. `dimacs`,`csr`,`el`).", 0},
    { "output_format", OPT_OUTPUT_FMT, "string",    0, "Force write to use this graph format (e.g. `dimacs`,`csr`,`el`).", 0},

    { "input",  OPT_INPUT_FILE,  "filename", 0, "Input filename.", 0},
    { "output", OPT_OUTPUT_FILE, "filename", 0, "Output filename.", 0},
    { "disk",   OPT_MMAP, "directory", OPTION_ARG_OPTIONAL, "Specify file template (ending with `XXXXXX`) to store intermediates in temporary files to reduce memory load. Working dir by default.", 0},

    { NULL, 0, NULL, 0, NULL, 0}
};

typedef struct cleaner_options {
    // Modification operations
    graph_size_t edge_alignment;
    bool to_directed;
    bool to_undirected;
    bool sort;

    // Clean operations
    bool transpose;
    bool unique;
    bool no_loops;
    bool connected;

    // Input / output handling
    char *input_file;
    char *input_ext;
    char *output_file;
    char *output_ext;
} cleaner_options_t;

static error_t argp_parser(int key, char *arg, struct argp_state *state) {
    cleaner_options_t *o = (cleaner_options_t*) state->input;

    switch (key)
    {
        case OPT_ALIGN_EDGES:   o->edge_alignment = sstrtoull(arg); break;
        case OPT_TO_DIRECTED:   o->to_directed    = strtob(arg, true); break;
        case OPT_TO_UNDIRECTED: o->to_undirected  = strtob(arg, true); break;
        case OPT_SORT:          o->sort           = strtob(arg, true); break;

        case OPT_TRANSPOSE: o->transpose = strtob(arg, true); break;
        case OPT_UNIQUE:    o->unique    = strtob(arg, true); break;
        case OPT_NO_LOOPS:  o->no_loops  = strtob(arg, true); break;
        case OPT_CONNECTED: o->connected = strtob(arg, true); break;

        case OPT_DIMACS:
        {
            o->no_loops    = true;
            o->to_directed = true;
            o->unique      = true;
            o->connected   = true;
            break;
        }

        case OPT_MMAP:
        {
            memory_set_default_pinned_manager(E_MM_MMAP);
            memory_set_mapped_tmpdir(arg);
            break;
        }

        case OPT_FORMAT:
        case OPT_INPUT_FMT:
        case OPT_OUTPUT_FMT:
        {
            if (key != OPT_OUTPUT_FMT)
            {
                if (o->input_ext == NULL)
                    o->input_ext = strdup(arg);
                else
                    return ARGP_ERR_UNKNOWN;
            }
            if (key != OPT_INPUT_FMT)
            {
                if (o->output_ext == NULL)
                    o->output_ext = strdup(arg);
                else
                    return ARGP_ERR_UNKNOWN;
            }
            break;
        }

        case OPT_INPUT_FILE:
        case OPT_OUTPUT_FILE:
        case ARGP_KEY_ARG:
        {
            if (key == OPT_INPUT_FILE || (key == ARGP_KEY_ARG && o->input_file == NULL))
                if (o->input_file == NULL)
                    o->input_file = strdup(arg);
                else
                    return ARGP_ERR_UNKNOWN;
            else if (o->output_file == NULL)
                o->output_file = strdup(arg);
            else
                return ARGP_ERR_UNKNOWN;
            break;
        }

        default:
            return ARGP_ERR_UNKNOWN;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    static cleaner_options_t options;

    const struct argp argp = {argp_options, argp_parser, argp_args_doc, argp_doc, NULL, NULL, NULL};
    if (argp_parse(&argp, argc, argv, 0, NULL, &options) != 0)
    {
        fprintf(stderr, "Could not parse application arguments.\n");
        return EXIT_FAILURE;
    }

    int res = EXIT_SUCCESS;

    cleaner_graph_t *graph = cleaner_graph(_read_file)(
        options.input_file,
        E_GRAPH_FLAG_PIN,
        options.input_file || options.input_ext ? options.input_ext : "el"
    );

    if (graph == NULL)
    {
        fprintf(stderr, "Invalid input `%s`!\n", options.input_file);
        res = EXIT_FAILURE;
    }
    else
    {
        fprintf(stderr, "G(|V|, |E|) = G(%zu, %zu)\n", (size_t)graph->vcount, (size_t)graph->ecount);

        if (options.no_loops)
            fprintf(stderr, "Removed %zu self loops.\n", (size_t)cleaner_graph(_remove_self_loops)(graph));

        if (options.to_directed)
        {
            cleaner_graph(_to_directed)(graph);
            fprintf(stderr, "Added %zu edges during undirected->directed conversion.\n", (size_t)graph->ecount / 2);
        }

        if (options.to_undirected)
            fprintf(stderr, "Removed %zu edges during directed->undirected conversion.\n", (size_t)cleaner_graph(_to_undirected)(graph));

        if (options.unique)
            fprintf(stderr, "Removed %zu duplicate edges.\n", (size_t)cleaner_graph(_remove_dup_edges)(graph, graph_merge_tag_add, NULL));

        if (options.connected)
            fprintf(stderr, "Removed %zu unconnected vertices.\n", (size_t)cleaner_graph(_remove_unconnected)(graph));

        if (options.transpose)
        {
            assert(!options.to_directed || options.to_undirected);
            cleaner_graph(_transpose)(graph);
            fprintf(stderr, "Reversed %zu edges.\n", (size_t)graph->ecount);
        }

        if (options.sort)
        {
            cleaner_graph_t *base = graph;
            graph = cleaner_graph(_sorted_copy)(base, base->flags);
            cleaner_graph(_free)(base);
            fprintf(stderr, "Sorted vertices by out-degree.\n");
        }

        if (options.edge_alignment > 1)
        {
            const graph_size_t v = cleaner_graph(_add_vertex)(graph);
            fprintf(stderr, "Added vertex (%zu) for edge alignment.\n", (size_t)v);
            fprintf(stderr, "Added %zu edges for alignment.\n", (size_t)cleaner_graph(_align_edges)(graph, options.edge_alignment, v));
        }

        if (graph->vcount > 0 && !cleaner_graph(_write_file)(
                graph,
                options.output_file,
                options.output_file || options.output_ext ? options.output_ext : "el"
           ))
        {
            fprintf(stderr, "Could not write to `%s`\n", options.output_file);
            res = EXIT_FAILURE;
        }

        cleaner_graph(_free)(graph);
    }

    free(options.input_file);
    free(options.input_ext);
    free(options.output_file);
    free(options.output_ext);

    return res;
}
