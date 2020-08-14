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

#define heat_graph_t   csr_graph_v_t
#define heat_graph(ID) csr_graph_v ## ID
#define heat_float_t   long double

static const char *const argp_doc      = "Output heat map for input graph";
static const char *const argp_args_doc = "[input_filename] [output_filename]";

const char *argp_program_version     = "ALPHA";
const char *argp_program_bug_address = "https://github.com/nielsAD";

enum HEATMAP_OPTIONS {
    OPT_BINS = 'b',

    OPT_MMAP        = 'D',
    OPT_FORMAT      = 'f',
    OPT_INPUT_FMT   = 'I',

    OPT_INPUT_FILE  = 'i',
    OPT_OUTPUT_FILE = 'o'
};

static const struct argp_option argp_options[] = {
    { "bins", OPT_BINS,  "integer", 0, "Number of row/col bins.", 0},

    { "input_format", OPT_INPUT_FMT, "string", 0, "Force reader to assume this graph format (e.g. `dimacs`,`csr`,`el`).", 0},
    { "format",       OPT_FORMAT,    "string", OPTION_ALIAS, NULL, 0},

    { "input",  OPT_INPUT_FILE,  "filename", 0, "Input filename.", 0},
    { "output", OPT_OUTPUT_FILE, "filename", 0, "Output filename.", 0},

    { "disk",   OPT_MMAP, "directory", OPTION_ARG_OPTIONAL, "Specify directory to store intermediates in temporary files to reduce memory load. Working dir by default.", 0},

    {NULL, 0, NULL, 0, NULL, 0}
};

typedef struct heatmap_options {
    // Heat map options
    graph_size_t bins;

    // Input / output handling
    char *input_file;
    char *input_ext;
    char *output_file;
} heatmap_options_t;

static error_t argp_parser(int key, char *arg, struct argp_state *state) {
    heatmap_options_t *o = (heatmap_options_t*) state->input;

    switch (key)
    {
        case OPT_BINS: o->bins = sstrtoull(arg); break;

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

heat_float_t *calculate_heatmap(const heat_graph_t *graph, const graph_size_t bins)
{
    heat_float_t *res = memory_talloc(heat_float_t, bins*bins);
    memset(res, 0, sizeof(*res) * bins*bins);

    const heat_float_t vcount = graph->vcount;
    csr_forall_edges(efr,eto,graph)
    {
        const size_t bfr = (efr / vcount) * bins;
        const size_t bto = (eto / vcount) * bins;
        res[(bfr*bins) + bto]++;
    }

    const heat_float_t mod = bins / vcount;
    for (size_t b = 0; b < bins*bins; b++)
    {
        res[b] *= mod;
    }

    return res;
}

static bool write_float_list(const heat_float_t *restrict arr, const graph_size_t len, const char *const name)
{
    assert(arr != NULL);

    FILE *file = (name == NULL) ? stdout : fopen(name, "w+");
    if (file == NULL)
        return false;

    for (graph_size_t i = 0; i < len; i++)
        fprintf(file, "%Lf\n", (long double)arr[i]);

    if (file != stdout)
        fclose(file);

    return true;
}

int main(int argc, char *argv[])
{
    static heatmap_options_t options;
    options.bins = 4096;

    const struct argp argp = {argp_options, argp_parser, argp_args_doc, argp_doc, NULL, NULL, NULL};
    if (argp_parse(&argp, argc, argv, 0, NULL, &options) != 0)
    {
        fprintf(stderr, "Could not parse application arguments.\n");
        return EXIT_FAILURE;
    }

    int res = EXIT_SUCCESS;

    heat_graph_t *graph = heat_graph(_read_file)(
        options.input_file,
        (graph_flags_enum_t) (E_GRAPH_FLAG_PIN),
        options.input_file || options.input_ext ? options.input_ext : "el"
    );

    if (graph == NULL || graph->vcount < 1 || graph->ecount < 1)
    {
        fprintf(stderr, "Invalid input file `%s`!\n", options.input_file);
        res = EXIT_FAILURE;
    }
    else
    {
        fprintf(stderr, "G(|V|, |E|) = G(%zu, %zu)\n", (size_t)graph->vcount, (size_t)graph->ecount);

        heat_float_t *arr = calculate_heatmap(graph, options.bins);
        if (!write_float_list(arr, options.bins*options.bins, options.output_file))
        {
            fprintf(stderr, "Could not write result to `%s`!\n", options.output_file);
            res = EXIT_FAILURE;
        }

        memory_free(arr);
        heat_graph(_free)(graph);
    }

    free(options.input_file);
    free(options.input_ext);
    free(options.output_file);

    return res;
}
