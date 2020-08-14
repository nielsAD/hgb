// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "util/string.h"
#include "util/file.h"

#include <argp.h>
#include <string>
#include <unordered_map>

static const char *const argp_doc      = "Maps unique string tokens to sequential numbers (based on order of appearance).";
static const char *const argp_args_doc = "[input_filename] [output_filename]";

const char *argp_program_version     = "ALPHA";
const char *argp_program_bug_address = "https://github.com/nielsAD";

enum CLEAN_OPTIONS {
    OPT_SKIP_LINES = 'l',
    OPT_SKIP_CHARS = 's',
    OPT_DELIMITERS = 'd',
    OPT_FIRST_NUM  = 'f',

    OPT_INPUT_FILE  = 'i',
    OPT_OUTPUT_FILE = 'o'
};

static const struct argp_option argp_options[] = {
    { "first", OPT_FIRST_NUM,  "num",   0, "Start output sequence at this number.", 0},
    { "line",  OPT_SKIP_LINES, "num",   0, "Skip the first `num` lines.", 0},
    { "skip",  OPT_SKIP_CHARS, "chars", 0, "Skip line if it starts with any of the characters in `chars`.", 0},
    { "delim", OPT_DELIMITERS, "chars", 0, "Explicitly specify set of delimiters for tokenization.", 0},

    { "input",  OPT_INPUT_FILE,  "filename", 0, "Input filename.", 0},
    { "output", OPT_OUTPUT_FILE, "filename", 0, "Output filename.", 0},
    { NULL, 0, NULL, 0, NULL, 0}
};

typedef struct strtoidx_options {
    size_t first_num;
    size_t skip_lines;

    char *skip_chars;
    char *delimiters;

    char *input_file;
    char *output_file;
} strtoidx_options_t;

static error_t argp_parser(int key, char *arg, struct argp_state *state) {
    strtoidx_options_t *o = (strtoidx_options_t*) state->input;

    switch (key)
    {
        case OPT_FIRST_NUM:  o->first_num  = strtoull(arg, NULL, 0); break;
        case OPT_SKIP_LINES: o->skip_lines = strtoull(arg, NULL, 0); break;
        case OPT_SKIP_CHARS: o->skip_chars = strdup(arg); break;
        case OPT_DELIMITERS: o->delimiters = strdup(arg); break;

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
    static strtoidx_options_t options;

    const struct argp argp = {argp_options, argp_parser, argp_args_doc, argp_doc, NULL, NULL, NULL};
    if (argp_parse(&argp, argc, argv, 0, NULL, &options) != 0)
    {
        fprintf(stderr, "Could not parse application arguments.\n");
        return EXIT_FAILURE;
    }

    const char *skip = (options.skip_chars != NULL)
        ? options.skip_chars
        : "#%";

    const char *del = (options.delimiters != NULL && strlen(options.delimiters) > 0)
        ? options.delimiters
        : " \f\n\r\t\v";

    FILE *input = (options.input_file == NULL || strcmp(options.input_file, "-") == 0)
        ? stdin
        : fopen(options.input_file, "r");

    FILE *output = (options.output_file == NULL || strcmp(options.output_file, "-") == 0)
        ? stdout
        : fopen(options.output_file, "w+");

    if (input != NULL && output != NULL)
    {
        char   *buf_txt = NULL;
        size_t  buf_len = 0;
        ssize_t line_len;

        std::unordered_map<std::string, size_t> map;
        size_t counter = options.first_num;

        while ((line_len = getline(&buf_txt, &buf_len, input)) != -1)
        {
            if (options.skip_lines > 0 || (line_len > 0 && strchr(skip, buf_txt[0]) != NULL))
            {
                if (options.skip_lines > 0)
                    options.skip_lines--;
                fputs(buf_txt, output);
                continue;
            }

            char *buf_pos = buf_txt;
            while(*buf_pos)
            {
                const size_t wspan = strspn(buf_pos, del);
                if (wspan > 0)
                {
                    fwrite(buf_pos, sizeof(char), wspan, output);
                    buf_pos += wspan;
                }

                const size_t tspan = strcspn(buf_pos, del);
                if (tspan > 0)
                {
                    auto index = map.emplace(std::string(buf_pos, tspan), counter);
                    buf_pos += tspan;
                    fprintf(output, "%zu", index.first->second);
                    if (index.second)
                        counter++;
                }
            }
        }

        if (buf_txt != NULL)
            free(buf_txt);
    }

    if (input  != NULL && input  != stdin)  fclose(input);
    if (output != NULL && output != stdout) fclose(output);
    free(options.skip_chars);
    free(options.delimiters);
    free(options.input_file);
    free(options.output_file);

    return EXIT_SUCCESS;
}
