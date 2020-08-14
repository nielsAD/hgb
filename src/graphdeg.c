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

#define degree_graph_t   csr_graph_v_t
#define degree_graph(ID) csr_graph_v ## ID

static const char *const argp_doc      = "Output vertex degrees for input graph";
static const char *const argp_args_doc = "[input_filename] [output_filename]";

const char *argp_program_version     = "ALPHA";
const char *argp_program_bug_address = "https://github.com/nielsAD";

enum DEGREE_OPTIONS {
    OPT_METHOD    = 'm',
    OPT_HISTOGRAM = 'h',
    OPT_SUMMARY   = 's',

    OPT_MMAP        = 'D',
    OPT_FORMAT      = 'f',
    OPT_INPUT_FMT   = 'I',

    OPT_INPUT_FILE  = 'i',
    OPT_OUTPUT_FILE = 'o'
};

static const struct argp_option argp_options[] = {
    { "method",    OPT_METHOD,    "['in','out','both']", 0,    "Degree counting method.", 0},
    { "histogram", OPT_HISTOGRAM, "bool", OPTION_ARG_OPTIONAL, "Output vertex count per degree rather than degree count per vertex (histogram).", 0},
    { "summary",   OPT_SUMMARY,   "bool", OPTION_ARG_OPTIONAL, "Print summary.", 0},

    { "input_format", OPT_INPUT_FMT, "string", 0, "Force reader to assume this graph format (e.g. `dimacs`,`csr`,`el`).", 0},
    { "format",       OPT_FORMAT,    "string", OPTION_ALIAS, NULL, 0},

    { "input",  OPT_INPUT_FILE,  "filename", 0, "Input filename.", 0},
    { "output", OPT_OUTPUT_FILE, "filename", 0, "Output filename.", 0},

    { "disk",   OPT_MMAP, "directory", OPTION_ARG_OPTIONAL, "Specify directory to store intermediates in temporary files to reduce memory load. Working dir by default.", 0},

    {NULL, 0, NULL, 0, NULL, 0}
};

typedef struct degree_options {
    // Partitioning options
    graph_flags_enum_t method;
    bool histogram;

    // Input / output handling
    char *input_file;
    char *input_ext;
    char *output_file;
    bool summary;
} degree_options_t;

static error_t argp_parser(int key, char *arg, struct argp_state *state) {
    degree_options_t *o = (degree_options_t*) state->input;

    switch (key)
    {
        case OPT_HISTOGRAM: o->histogram = strtob(arg, true); break;
        case OPT_SUMMARY:   o->summary   = strtob(arg, true); break;

        case OPT_METHOD:
        {
            if (strcasecmp(arg, "in") == 0 || strcasecmp(arg, "i") == 0)
                o->method = E_GRAPH_FLAG_DEG_I;
            else if (strcasecmp(arg, "out") == 0 || strcasecmp(arg, "o") == 0)
                o->method = E_GRAPH_FLAG_DEG_O;
            else if (strcasecmp(arg, "inout") == 0 || strcasecmp(arg, "both") == 0 || strcasecmp(arg, "b") == 0 || strcasecmp(arg, "io") == 0)
                o->method = E_GRAPH_FLAG_DEG_IO;
            else
                return ARGP_ERR_UNKNOWN;
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

graph_size_t *calculate_histogram(const graph_size_t *arr, const graph_size_t len, graph_size_t *res_size)
{
    graph_size_t max = 0;

    if (res_size == NULL || *res_size <= 1)
    {
        OMP_PRAGMA(omp parallel for reduction(max:max))
        for (graph_size_t i = 0; i < len; i++)
            if (arr[i] >= max)
                max = arr[i]+1;

        if (res_size == NULL)
            res_size = &max;
        if (*res_size <= 1)
            *res_size = max;
    }

    max = *res_size;
    graph_size_t *res = memory_talloc(graph_size_t, max);
    memset(res, 0, sizeof(*res) * max);

    for (graph_size_t i = 0; i < len; i++)
        if (arr[i] < max)
            res[arr[i]]++;

    return res;
}

static int _sort_int(const void *a, const void *b)
{
    return *(graph_size_t*)a - *(graph_size_t*)b;
}

static void print_summary(const graph_size_t *restrict arr, const graph_size_t len)
{
    qsort((void*)arr, len, sizeof(graph_size_t), _sort_int);

    long double avg = 0.0;
    OMP_PRAGMA(omp parallel for reduction(+:avg))
    for (graph_size_t i = 0; i < len; i++)
    {
        avg += arr[i];
    }
    avg /= len;

    long double stddev = 0.0;
    OMP_PRAGMA(omp parallel for reduction(+:stddev))
    for (graph_size_t i = 0; i < len; i++)
    {
        const long double d = arr[i] - avg;
        stddev += d*d;
    }
    stddev = sqrtl(stddev / len);

    const long double skewness = ((long double)(3.0) * (avg - arr[len / 2])) / stddev;
    printf("avg=%.3Lf stddev=%.3Lf skewness=%.3Lf\n", avg, stddev, skewness);
    printf("P0=%u P5=%u P10=%u P25=%u P50=%u P75=%u P90=%u P95=%u P100=%u\n",
        arr[0],
        arr[len / 20],
        arr[len / 10],
        arr[len / 4],
        arr[len / 2],
        arr[len - 1 - (len / 4)],
        arr[len - 1 - (len / 10)],
        arr[len - 1 - (len / 20)],
        arr[len - 1]
    );
}

static bool write_integer_list(const graph_size_t *restrict arr, const graph_size_t len, const char *const name)
{
    assert(arr != NULL);

    FILE *file = (name == NULL) ? stdout : fopen(name, "w+");
    if (file == NULL)
        return false;

    for (graph_size_t i = 0; i < len; i++)
        fprintf(file, "%zu\n", (size_t)arr[i]);

    if (file != stdout)
        fclose(file);

    return true;
}

int main(int argc, char *argv[])
{
    static degree_options_t options;
    options.method = E_GRAPH_FLAG_DEG_IO;

    const struct argp argp = {argp_options, argp_parser, argp_args_doc, argp_doc, NULL, NULL, NULL};
    if (argp_parse(&argp, argc, argv, 0, NULL, &options) != 0)
    {
        fprintf(stderr, "Could not parse application arguments.\n");
        return EXIT_FAILURE;
    }

    int res = EXIT_SUCCESS;

    degree_graph_t *graph = degree_graph(_read_file)(
        options.input_file,
        (graph_flags_enum_t) (E_GRAPH_FLAG_PIN | options.method),
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

        graph_size_t  len = 0;
        graph_size_t *deg;
        graph_size_t *arr;

        switch(options.method)
        {
            case E_GRAPH_FLAG_DEG_I:  deg = (graph_size_t*) degree_graph(_get_vertex_degrees_in)(graph);  break;
            case E_GRAPH_FLAG_DEG_O:  deg = (graph_size_t*) degree_graph(_get_vertex_degrees_out)(graph); break;
            case E_GRAPH_FLAG_DEG_IO: deg = (graph_size_t*) degree_graph(_get_vertex_degrees)(graph);     break;
            default:
                fprintf(stderr, "Invalid degree counting method\n");
                res = EXIT_FAILURE;
                goto finalize;
        }

        if (options.histogram)
            arr = calculate_histogram(deg, graph->vcount, &len);
        else
        {
            len = graph->vcount;
            arr = deg;
        }

        if (!write_integer_list(arr, len, options.output_file))
        {
            fprintf(stderr, "Could not write result to `%s`!\n", options.output_file);
            res = EXIT_FAILURE;
        }

        if (options.summary)
        {
            print_summary(arr, len);
        }

        if (arr != deg)                            memory_free(arr);
        if (options.method == E_GRAPH_FLAG_DEG_IO) memory_free(deg);

        finalize:
            degree_graph(_free)(graph);
    }

    free(options.input_file);
    free(options.input_ext);
    free(options.output_file);

    return res;
}
