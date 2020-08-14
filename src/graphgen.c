// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "util/math.h"
#include "util/memory.h"
#include "util/file.h"
#include "util/string.h"
#include "util/igraph.h"

#include <argp.h>
#include <time.h>

#define output_graph_t   igraph_csr_graph_t
#define output_graph(ID) igraph_csr_graph(ID)

const graph_flags_enum_t GRAPH_FLAGS = E_GRAPH_FLAG_SORT;

static const char *const argp_doc      = "Graph generator using selectable algorithm";
static const char *const argp_args_doc = "numvertices [output_filename]";

const char *argp_program_version     = "ALPHA";
const char *argp_program_bug_address = "https://github.com/nielsAD";

enum generator_cl_options {
    OPT_ALGORITHM = 'a',
    OPT_VERTICES  = 'v',
    OPT_EDGES     = 'e',
    OPT_EDGES_MOD = 'm',
    OPT_RAND_SEED = 'r',

    OPT_MMAP        = 'D',
    OPT_FORMAT      = 'f',
    OPT_OUTPUT_FMT  = 'O',
    OPT_OUTPUT_FILE = 'o'
};

static const struct argp_option argp_options[] = {
    { "algo",   OPT_ALGORITHM, "alg", 0, "Generation algorithm.", 0},
    { "nvert",  OPT_VERTICES,  "num", 0, "Number of vertices.", 0},
    { "nedge",  OPT_EDGES,     "num", 0, "Number of edges.", 0},
    { "medge",  OPT_EDGES_MOD, "num", 0, "Number of edges modifier (multiplied by number of vertices).", 0},
    { "random", OPT_RAND_SEED, "int", 0, "Random seed.", 0},

    { "output_format", OPT_OUTPUT_FMT, "string", 0, "Force writer to assume this graph format (e.g. `dimacs`,`csr`,`el`).", 0},
    { "format",        OPT_FORMAT,     "string", OPTION_ALIAS, NULL, 0},

    { "output", OPT_OUTPUT_FILE, "filename", 0, "Output filename.", 0},
    { "disk",   OPT_MMAP, "directory", OPTION_ARG_OPTIONAL, "Specify directory to store intermediates in temporary files to reduce memory load. Working dir by default.", 0},

    {NULL, 0, NULL, 0, NULL, 0}
};

typedef enum generator {
    GEN_RANDOM,
    GEN_RANDOM_GRG,
    GEN_BARABASI,
    GEN_FOREST_FIRE,
    GEN_POWERLAW,
    GEN_KRONECKER,
    GEN_REGULAR,
    GEN_LATTICE,
    GEN_STAR,
    GEN_TREE,
    GEN_FULL
} generator_enum_t;

typedef struct generator_options {
    // Partitioning options
    generator_enum_t method;
    graph_size_t v;
    graph_size_t e;
    graph_size_t emod;

    // Output handling
    char *output_file;
    char *output_ext;
} generator_options_t;

static error_t argp_parser(int key, char *arg, struct argp_state *state) {
    generator_options_t *o = (generator_options_t*) state->input;

    switch (key)
    {
        case OPT_VERTICES:  o->v    = sstrtoull(arg); break;
        case OPT_EDGES:     o->e    = sstrtoull(arg); break;
        case OPT_EDGES_MOD: o->emod = sstrtoull(arg); break;

        case OPT_RAND_SEED: srand((unsigned int)(sstrtoull(arg))); break;

        case OPT_ALGORITHM:
        {
            if (strcasecmp(arg, "random") == 0 || strcasecmp(arg, "rnd") == 0 || strcasecmp(arg, "erdos_renyi") == 0 || strcasecmp(arg, "erdosrenyi") == 0 || strcasecmp(arg, "er") == 0 || strcasecmp(arg, "r") == 0)
                o->method = GEN_RANDOM;
            else if (strcasecmp(arg, "grg_random") == 0 || strcasecmp(arg, "grgrandom") == 0 || strcasecmp(arg, "grg") == 0 || strcasecmp(arg, "g") == 0)
                o->method = GEN_RANDOM_GRG;
            else if (strcasecmp(arg, "barabasi") == 0 || strcasecmp(arg, "brb") == 0 || strcasecmp(arg, "b") == 0)
                o->method = GEN_BARABASI;
            else if (strcasecmp(arg, "forest_fire") == 0 || strcasecmp(arg, "ff") == 0)
                o->method = GEN_FOREST_FIRE;
            else if (strcasecmp(arg, "powerlaw") == 0 || strcasecmp(arg, "pl") == 0 || strcasecmp(arg, "p") == 0)
                o->method = GEN_POWERLAW;
            else if (strcasecmp(arg, "kronecker") == 0 || strcasecmp(arg, "k") == 0 || strcasecmp(arg, "graph500") == 0)
                o->method = GEN_KRONECKER;
            else if (strcasecmp(arg, "regular") == 0 || strcasecmp(arg, "reg") == 0)
                o->method = GEN_REGULAR;
            else if (strcasecmp(arg, "lattice") == 0 || strcasecmp(arg, "l") == 0)
                o->method = GEN_LATTICE;
            else if (strcasecmp(arg, "star") == 0 || strcasecmp(arg, "s") == 0)
                o->method = GEN_STAR;
            else if (strcasecmp(arg, "tree") == 0 || strcasecmp(arg, "t") == 0)
                o->method = GEN_TREE;
            else if (strcasecmp(arg, "full") == 0 || strcasecmp(arg, "f") == 0)
                o->method = GEN_FULL;
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
        case OPT_OUTPUT_FMT:
        {
            if (o->output_ext == NULL)
                o->output_ext = strdup(arg);
            else
                return ARGP_ERR_UNKNOWN;
            break;
        }

        case OPT_OUTPUT_FILE:
        case ARGP_KEY_ARG:
        {
            if (key == ARGP_KEY_ARG && o->v < 1)
                o->v = sstrtoull(arg);
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
    static generator_options_t options;
    srand(time(NULL));

    const struct argp argp = {argp_options, argp_parser, argp_args_doc, argp_doc, NULL, NULL, NULL};
    if (argp_parse(&argp, argc, argv, 0, NULL, &options) != 0 || options.v < 1)
    {
        fprintf(stderr, "Could not parse application arguments.\n");
        return EXIT_FAILURE;
    }

    options.e += options.v * options.emod;

    int res = EXIT_SUCCESS;

    igraph_t        g;
    output_graph_t *o = NULL;

    igraph_empty(&g, 0, true);

    switch(options.method)
    {
        case GEN_RANDOM:
        {
            o = output_graph(_new_random)(options.v, options.e, GRAPH_FLAGS);
            break;
        }

        case GEN_REGULAR:
        {
            o = output_graph(_new_regular)(options.v, options.e / options.v, 1, GRAPH_FLAGS);
            break;
        }

        case GEN_KRONECKER:
        {
            o = output_graph(_new_kronecker)(options.v, options.e, GRAPH_FLAGS);
            break;
        }

        case GEN_RANDOM_GRG:
        {
            const igraph_real_t r = sqrt((options.e * 2) / (options.v * options.v * M_PI));
            igraph_grg_game(&g, options.v, r, true, NULL, NULL);
            break;
        }

        case GEN_BARABASI:
        {
            const igraph_barabasi_algorithm_t mode = (options.v > options.e)
                ? IGRAPH_BARABASI_PSUMTREE_MULTIPLE
                : IGRAPH_BARABASI_PSUMTREE;
            igraph_barabasi_game(&g, options.v, 2.5, options.e / options.v, NULL, true, 1, true, mode, NULL);
            break;
        }

        case GEN_FOREST_FIRE:
        {
            igraph_forest_fire_game(&g, options.v, 0.1, 0.15, options.e / options.v, true);
            break;
        }

        case GEN_POWERLAW:
        {
            igraph_static_power_law_game(&g, options.v, options.e, 2.5, 2.5, false, options.v > options.e, true);
            break;
        }

        case GEN_LATTICE:
        {
            bool mutual = round(options.e / (double)options.v) > 1;
            if (mutual) options.e /= 2;

            igraph_vector_t v;
            igraph_vector_init(&v, ceil(options.e / 2.0 / (double) options.v));
            igraph_vector_fill(&v, ceil(pow(options.v, 1.0 / igraph_vector_size(&v))));
            igraph_lattice(&g, &v, 1, mutual, true, true);
            igraph_vector_destroy(&v);
            break;
        }

        case GEN_STAR:
        {
            const igraph_star_mode_t mode = round(options.e / (double) options.v) > 1
                ? IGRAPH_STAR_MUTUAL
                : IGRAPH_STAR_IN;
            igraph_star(&g, options.v, mode, 0);
            break;
        }

        case GEN_TREE:
        {
            igraph_tree(&g, options.v, ceil(options.e / (double) options.v), IGRAPH_TREE_IN);
            break;
        }

        case GEN_FULL:
        {
            igraph_full(&g, sqrt(MAX(options.e, options.v)), true, false);
            break;
        }
    }

    if (igraph_vcount(&g) > 0)
    {
        assert(o == NULL);
        o = output_graph(_from_igraph)(&g, GRAPH_FLAGS);
    }

    igraph_destroy(&g);

    if (o == NULL || options.v == 0 || !output_graph(_write_file)(
            o,
            options.output_file,
            options.output_file || options.output_ext ? options.output_ext : "el"
       ))
    {
        fprintf(stderr, "Could not write to `%s`\n", options.output_file);
        res = EXIT_FAILURE;
    }

    output_graph(_free)(o);
    free(options.output_file);
    free(options.output_ext);

    return res;
}
