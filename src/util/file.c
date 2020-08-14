// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "util/file.h"
#include "util/string.h"

unsigned long long int line_count(FILE *stream)
{
    fpos_t pos;
    if (fgetpos(stream, &pos) != 0)
        return 0;

    unsigned long long int res = 1;

    int c;
    while (EOF != (c = fgetc(stream)))
        if (c =='\n')
            ++res;

    fsetpos(stream, &pos);
    return res;
}

const char *file_extension(const char *const filename)
{
    const char* base = basename(filename);
    const char* ext = strrchr(base, '.');
    return (ext) ? ext + 1 : NULL;
}
