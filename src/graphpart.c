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
#include <time.h>

#define partitioner_graph_t   csr_graph_v_t
#define partitioner_graph(ID) csr_graph_v ## ID

static const char *const argp_doc      = "Distributes input graph over several partitions.";
static const char *const argp_args_doc = "num_parts output_filename [input_filename]";

const char *argp_program_version     = "ALPHA";
const char *argp_program_bug_address = "https://github.com/nielsAD";

enum PARTITIONER_OPTIONS {
    OPT_METHOD = 'm',
    OPT_CROSS_MERGE = 'n',

    OPT_INPUT_PARTS       = 'p',
    OPT_OUTPUT_INDEX      = 'x',
    OPT_OUTPUT_SUBGRAPHS  = 's',
    OPT_OUTPUT_CROSSGRAPH = 'c',

    OPT_MMAP        = 'D',
    OPT_FORMAT      = 'f',
    OPT_INPUT_FMT   = 'I',
    OPT_OUTPUT_FMT  = 'O',

    OPT_RAND_SEED   = 'r',
    OPT_INPUT_FILE  = 'i',
    OPT_OUTPUT_FILE = 'o'
};

typedef enum partitioner_method {
    E_PM_BLOCK,
    E_PM_RANDOM,
    E_PM_FILE,
} partitioner_method_t;

static const struct argp_option argp_options[] = {
    { "method",      OPT_METHOD, "['block','random','file']", 0, "Graph partitioning strategy. Blocks by default.", 0},
    { "input_parts", OPT_INPUT_PARTS, "filename",             0, "File which indicates partition per vector. Implies method=file.", 0},

    { "no_merge",      OPT_CROSS_MERGE,       NULL, OPTION_ARG_OPTIONAL, "Do not merge crossing edges with the same destination.", 0},
    { "no_index",      OPT_OUTPUT_INDEX,      NULL, OPTION_ARG_OPTIONAL, "Output index file containing partition and index for each vertex.", 0},
    { "no_subgraph",   OPT_OUTPUT_SUBGRAPHS,  NULL, OPTION_ARG_OPTIONAL, "Output graph files containing partition subgraphs.", 0},
    { "no_crossgraph", OPT_OUTPUT_CROSSGRAPH, NULL, OPTION_ARG_OPTIONAL, "Output graph containing the crossing edges between partitions.", 0},

    { "format",        OPT_FORMAT,     "string",    0, "Force reader/writer to use this format (e.g. `dimacs`,`csr`,`el`).", 0},
    { "input_format",  OPT_INPUT_FMT,  "string",    0, "Force reader to assume this graph format (e.g. `dimacs`,`csr`,`el`).", 0},
    { "output_format", OPT_OUTPUT_FMT, "string",    0, "Force write to use this graph format (e.g. `dimacs`,`csr`,`el`).", 0},

    { "input",  OPT_INPUT_FILE,  "filename", 0, "Input filename.", 0},
    { "output", OPT_OUTPUT_FILE, "filename", 0, "Output filename base. Is appended suffix depending on result file.", 0},

    { "disk",   OPT_MMAP, "directory", OPTION_ARG_OPTIONAL, "Specify directory to store intermediates in temporary files to reduce memory load. Working dir by default.", 0},
    { "random", OPT_RAND_SEED,  "int", 0, "Random seed.", 0},

    {NULL, 0, NULL, 0, NULL, 0}
};

typedef struct partitioner_options {
    // Partitioning options
    graph_size_t num_parts;
    partitioner_method_t method;
    bool no_merge;

    // Output options
    bool no_index;
    bool no_subgraphs;
    bool no_crossgraph;

    // Input / output handling
    char *input_parts;
    char *input_file;
    char *input_ext;
    char *output_file;
    char *output_ext;
} partitioner_options_t;

typedef struct graph_partitioning {
    graph_size_t part;
    graph_size_t vcount;
    graph_size_t *vertex_part;
    graph_size_t *vertex_part_index;
    pcsr_cross_graph_t *cross_graph;
} graph_partitioning_t;

static bool partition_cross_mapping_fun_e(const graph_size_t old_src, const graph_size_t old_dst, graph_size_t *restrict new_src, graph_size_t *restrict new_dst, graph_size_t *restrict part)
{
    assert(new_src != NULL);
    assert(new_dst != NULL);
    assert(part != NULL);

    if (part[old_src] == part[old_dst])
        return false;

    *new_src = old_dst;
    *new_dst = part[old_src];

    return true;
}

static bool partition_mapping_fun_v(const graph_size_t old_index, graph_size_t *restrict new_index, graph_partitioning_t *restrict args)
{
    assert(new_index != NULL);
    assert(args != NULL);

    graph_size_t eid;

    if (args->vertex_part[old_index] == args->part)
    {
        *new_index = args->vertex_part_index[old_index];
        return true;
    } else if (pcsr_cross_graph(_get_edge_index)(args->cross_graph, old_index, args->part, &eid))
    {
        *new_index = pcsr_cross_graph(_get_edge_tag)(args->cross_graph, eid);
        return true;
    }
    else
        return false;
}

static bool partition_mapping_fun_e(const graph_size_t old_src, const graph_size_t old_dst, graph_size_t *restrict new_src, graph_size_t *restrict new_dst, graph_partitioning_t *restrict args)
{
    assert(new_src != NULL);
    assert(new_dst != NULL);
    assert(args != NULL);

    if (args->vertex_part[old_src] != args->part)
        return false;
    *new_src = args->vertex_part_index[old_src];

    UNUSED bool res = partition_mapping_fun_v(old_dst, new_dst, args);
    assert(res);

    return true;
}

static error_t argp_parser(int key, char *arg, struct argp_state *state) {
    partitioner_options_t *o = (partitioner_options_t*) state->input;

    switch (key)
    {
        case OPT_CROSS_MERGE:       o->no_merge      = strtob(arg, true); break;
        case OPT_OUTPUT_INDEX:      o->no_index      = strtob(arg, true); break;
        case OPT_OUTPUT_SUBGRAPHS:  o->no_subgraphs  = strtob(arg, true); break;
        case OPT_OUTPUT_CROSSGRAPH: o->no_crossgraph = strtob(arg, true); break;

        case OPT_RAND_SEED: srand((unsigned int)(sstrtoull(arg))); break;

        case OPT_INPUT_PARTS:
        {
            o->input_parts = strdup(arg);
            o->method      = E_PM_FILE;
            o->no_index    = true;
            break;
        }

        case OPT_METHOD:
        {
            if (strcasecmp(arg, "block") == 0 || strcasecmp(arg, "blocks") == 0 || strcasecmp(arg, "b") == 0)
                o->method = E_PM_BLOCK;
            else if (strcasecmp(arg, "random") == 0 || strcasecmp(arg, "rnd") == 0 || strcasecmp(arg, "r") == 0)
                o->method = E_PM_RANDOM;
            else if (strcasecmp(arg, "file") == 0 || strcasecmp(arg, "f") == 0)
            {
                o->method   = E_PM_FILE;
                o->no_index = true;
            }
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

        case ARGP_KEY_ARG:
            if (o->num_parts < 1)
            {
                o->num_parts = sstrtoull(arg);
                if (o->num_parts > 0)
                    break;
                else
                    return ARGP_ERR_UNKNOWN;
            }

            fallthrough;

        case OPT_INPUT_FILE:
        case OPT_OUTPUT_FILE:
        {
            if (key == OPT_OUTPUT_FILE || (key == ARGP_KEY_ARG && o->output_file == NULL))
                if (o->output_file == NULL)
                    o->output_file = strdup(arg);
                else
                    return ARGP_ERR_UNKNOWN;
            else if (o->input_file == NULL)
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

static bool graph_partition_blocks(graph_size_t *restrict res, const graph_size_t vcount, const graph_size_t num_parts)
{
    assert(res != NULL);
    assert(num_parts > 0);

    const graph_size_t bsize = (vcount + num_parts - 1) / num_parts;

    OMP_PRAGMA(omp parallel for)
    for (graph_size_t v = 0; v < vcount; v++)
        res[v] = v / bsize;

    return true;
}

static bool graph_partition_random(graph_size_t *restrict res, const graph_size_t vcount, const graph_size_t num_parts)
{
    assert(res != NULL);
    assert(num_parts <= RAND_MAX);

    unsigned int seed[omp_get_max_threads()];
    for (int t = 0; t < omp_get_max_threads(); t++)
        seed[t] = rand();

    OMP_PRAGMA(omp parallel)
    {
        const int thread_num = omp_get_thread_num();

        OMP_PRAGMA(omp for)
        for (graph_size_t v = 0; v < vcount; v++)
            res[v] = rand_r(&seed[thread_num]) % num_parts;
    }

    return true;
}

static bool graph_partition_file(graph_size_t *restrict res, const graph_size_t vcount, const char *const name)
{
    FILE* file = (name == NULL || strcmp(name, "-") == 0)
        ? stdin
        : fopen(name, "r");

    if (file == NULL)
        return false;

    char  *buf_txt = NULL;
    size_t buf_len = 0;
    ssize_t line_len;

    graph_size_t v = 0;
    while (v < vcount && (line_len = getline(&buf_txt, &buf_len, file)) != -1)
        res[v++] = strtoull(buf_txt, NULL, 0);

    free(buf_txt);
    if (file != stdin)
        fclose(file);

    return v >= vcount;
}

static bool write_graph_partitioning_index(const graph_partitioning_t *restrict part, const char *const base)
{
    assert(part != NULL);

    char *name = (base == NULL || strcmp(base, "-") == 0)
        ? NULL
        : pcsr_graph_filename_index(base, part->part);
    FILE *file = (name == NULL) ? stdout : fopen(name, "w+");

    if (file == NULL)
    {
        fprintf(stderr, "Could not write vertex index to `%s`!\n", name);
        free(name);
        return false;
    } else if (name != NULL)
    {
        fprintf(stderr, "Vertex index -> %s\n", name);
        free(name);
    }

    for (graph_size_t v = 0; v < part->vcount; v++)
    {
        fprintf(file, "%zu %zu", (size_t)part->vertex_part[v], (size_t)part->vertex_part_index[v]);
        csr_forall_out_edges(e, p, v, part->cross_graph)
        {
            fprintf(file, " %zu %zu",
                (size_t)p,                             // Partition
                (size_t)part->cross_graph->etag[e]     // Index in partition
            );
        }
        fputc('\n', file);
    }

    if (file != stdout)
        fclose(file);

    return true;
}

static bool write_partitioning_crossgraph(const graph_partitioning_t *restrict part, const char *const base)
{
    pcsr_cross_graph_eli_t *rep = pcsr_cross_graph(_get_eli_representation)(part->cross_graph);
    if (rep == NULL)
        return false;

    pcsr_cross_graph_eli_t *eli = pcsr_cross_graph_eli(_copy)(rep, rep->flags);
    pcsr_cross_graph(_free_eli_representation)(part->cross_graph, rep);

    if (eli == NULL)
        return false;

    eli_forall_edges_par(e, eli, /*no omp params*/)
    {
        const graph_size_t part_fr = part->vertex_part[eli->efr[e]];
        const graph_size_t part_to = eli->eto[e];

        eli->efr[e] = part->vertex_part_index[eli->efr[e]];
        eli->eto[e] = eli->etag[e]; // == vertex_part_index[original_dst]

        eli->etag[e] = pcsr_cross_graph_tag_encode_parts(part_fr, part_to);
    }

    pcsr_cross_graph_sort_eli(eli);

    char *name = (base == NULL || strcmp(base, "-") == 0)
        ? NULL
        : pcsr_graph_filename_crossgraph(base, part->part);

    const bool res = pcsr_cross_graph_eli(_write_file)(eli, name, "el_gz");

    if (!res)
        fprintf(stderr, "Could not write crossgraph to `%s`\n", name);
    else if (name != NULL)
        fprintf(stderr, "Cross graph -> %s\n", name);

    free(name);
    pcsr_cross_graph_eli(_free)(eli);

    return res;
}

static bool write_graph_partitions(const graph_partitioning_t *restrict part, const partitioner_graph_t *restrict graph, const char *const base, const char *const ext)
{
    bool res = true;

    OMP_PRAGMA(omp parallel for)
    for (graph_size_t s = 0; s < part->part; s++)
    {
        graph_partitioning_t arg = *part;
        arg.part = s;

        partitioner_graph_t *sub = partitioner_graph(_mapped_copy)(
            graph,
            graph->flags,
            (graph_map_vertex_func_t)partition_mapping_fun_v,
            (graph_map_edge_func_t)partition_mapping_fun_e,
            &arg
        );

        if (sub == NULL)
        {
            fprintf(stderr, "Could not map subgraph %zu!\n", (size_t)s);
            res = false;
            continue;
        }

        char *name = (base == NULL || strcmp(base, "-") == 0)
            ? NULL
            : pcsr_graph_filename_partition(base, part->part, s);

        if (partitioner_graph(_write_file)(
            sub,
            name,
            name || ext ? ext : "el"))
        {
            fprintf(stderr, "S%zu(|V|, |E|) = G(%zu, %zu) -> %s\n", (size_t)s, (size_t)sub->vcount, (size_t)sub->ecount, name);
        }
        else
        {
            fprintf(stderr, "Could not write subgraph %zu to `%s`\n", (size_t)s, name);
            res = false;
        }

        free(name);
        partitioner_graph(_free)(sub);
    }

    return res;
}

int main(int argc, char *argv[])
{
    static partitioner_options_t options;
    options.method = E_PM_BLOCK;
    srand(time(NULL));

    const struct argp argp = {argp_options, argp_parser, argp_args_doc, argp_doc, NULL, NULL, NULL};
    if (argp_parse(&argp, argc, argv, 0, NULL, &options) != 0 || options.num_parts < 1 || options.output_file == NULL)
    {
        fprintf(stderr, "Could not parse application arguments.\n");
        return EXIT_FAILURE;
    }

    int res = EXIT_SUCCESS;

    partitioner_graph_t *graph = partitioner_graph(_read_file)(
        options.input_file ? options.input_file : options.output_file,
        E_GRAPH_FLAG_PIN,
        options.input_ext
    );

    if (graph == NULL || graph->vcount < 1 || graph->ecount < 1)
    {
        fprintf(stderr, "Invalid input file `%s`!\n", options.input_file);
        res = EXIT_FAILURE;
    }
    else
    {

        fprintf(stderr, "G(|V|, |E|) = G(%zu, %zu)\n", (size_t)graph->vcount, (size_t)graph->ecount);

        graph_partitioning_t part = {
            options.num_parts,
            graph->vcount,
            memory_pinned_talloc(graph_size_t, graph->vcount),
            memory_pinned_talloc(graph_size_t, graph->vcount),
            NULL
        };

        bool parted = false;
        graph_size_t local_vcount[options.num_parts];
        memset(local_vcount, 0, sizeof(local_vcount));

        if (options.input_parts == NULL && (options.input_file || options.output_file))
            options.input_parts = pcsr_graph_filename_index(
                options.input_file ? options.input_file : options.output_file,
                options.num_parts
            );

        switch (options.method)
        {
            case E_PM_BLOCK:  parted = graph_partition_blocks(part.vertex_part, part.vcount, options.num_parts); break;
            case E_PM_RANDOM: parted = graph_partition_random(part.vertex_part, part.vcount, options.num_parts); break;
            case E_PM_FILE:   parted = graph_partition_file(part.vertex_part, part.vcount, options.input_parts ? options.input_parts : options.input_file); break;
        }

        if (!parted)
        {
            fprintf(stderr, "Could not divide part input graph!\n");
            res = EXIT_FAILURE;
            goto finalize;
        }

        for (graph_size_t v = 0; v < part.vcount; v++)
            part.vertex_part_index[v] = local_vcount[part.vertex_part[v]]++;

        part.cross_graph = pcsr_cross_graph(_mapped_copy)(
            graph,
            (graph_flags_enum_t) (E_GRAPH_FLAG_PIN | E_GRAPH_FLAG_SORT),
            graph_map_vertex_noop,
            (graph_map_edge_func_t)partition_cross_mapping_fun_e,
            part.vertex_part
        );
        if (part.cross_graph == NULL)
        {
            fprintf(stderr, "Could not map temporary cross graph!\n");
            res = EXIT_FAILURE;
            goto finalize;
        }

        if (!options.no_merge)
            pcsr_cross_graph(_remove_dup_edges)(part.cross_graph, NULL, NULL);
        pcsr_cross_graph(_toggle_flag)(part.cross_graph, E_GRAPH_FLAG_TAG_E, true);

        csr_forall_edges(e, part.cross_graph)
        {
            part.cross_graph->etag[e] = local_vcount[part.cross_graph->col_idx[e]]++;
        }

        fprintf(stderr, "%zu (%.0f%%) crossing edges\n", (size_t)part.cross_graph->ecount, part.cross_graph->ecount / (double)graph->ecount * 100.0);

        if (!options.no_index && !write_graph_partitioning_index(&part, options.output_file))
            res = EXIT_FAILURE;

        if (!options.no_subgraphs && !write_graph_partitions(&part, graph, options.output_file, options.output_ext))
            res = EXIT_FAILURE;

        if (!options.no_crossgraph && !write_partitioning_crossgraph(&part, options.output_file))
            res = EXIT_FAILURE;

        finalize:
            memory_pinned_free(part.vertex_part);
            memory_pinned_free(part.vertex_part_index);
            if (part.cross_graph  != NULL) pcsr_cross_graph(_free)(part.cross_graph);
            partitioner_graph(_free)(graph);
    }

    free(options.input_parts);
    free(options.input_file);
    free(options.input_ext);
    free(options.output_file);
    free(options.output_ext);

    return res;
}
