// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "util/string.h"

bool strtob(const char *const a, const bool def)
{
    return (a == NULL)
        ? def
        :  strcasecmp("0", a)     != 0
        && strcasecmp("f", a)     != 0
        && strcasecmp("false", a) != 0
        && strcasecmp("n", a)     != 0
        && strcasecmp("no", a)    != 0;
}

unsigned long long int sstrtoull(const char *const a)
{
    char *e;
    unsigned long long int res = strtoull(a, &e, 0);

    switch(*e)
    {
        case 'k':
        case 'K': res *= 1024LL; break;
        case 'm':
        case 'M': res *= 1024LL*1024; break;
        case 'g':
        case 'G': res *= 1024LL*1024*1024; break;
        case 't':
        case 'T': res *= 1024LL*1024*1024*1024; break;
        case 'p':
        case 'P': res *= 1024LL*1024*1024*1024*1024; break;
        case 'e':
        case 'E': res *= 1024LL*1024*1024*1024*1024*1024; break;
        //case 'z':
        //case 'Z': res *= 1024LL*1024*1024*1024*1024*1024*1024; break;
    }

    return res;
}

unsigned long long int strtoullr(const char *const a, const unsigned long long int min, const unsigned long long int max)
{
    if (
         strcasecmp("min", a)   == 0
      || strcasecmp("first", a) == 0
      || strcasecmp("begin", a) == 0
      || strcasecmp("b", a)     == 0
    )
        return min;
    else if (
         strcasecmp("max", a)  == 0
      || strcasecmp("last", a) == 0
      || strcasecmp("end", a)  == 0
      || strcasecmp("e", a)    == 0
    )
        return max;
    else
        return sstrtoull(a);
}

// http://creativeandcritical.net/str-replace-c
char *strrepl(const char *str, const char *pat, const char *repl)
{
    /* Increment positions cache size initially by this number. */
    size_t cache_sz_inc = 16;
    /* Thereafter, each time capacity needs to be increased,
     * multiply the increment by this factor. */
    const size_t cache_sz_inc_factor = 3;
    /* But never increment capacity by more than this number. */
    const size_t cache_sz_inc_max = 1048576;

    char *pret, *ret = NULL;
    const char *pstr2, *pstr = str;
    size_t i, count = 0;
    ptrdiff_t *pos_cache = NULL;
    size_t cache_sz = 0;
    size_t cpylen, orglen, retlen, repllen, patlen = strlen(pat);

    /* Find all matches and cache their positions. */
    while ((pstr2 = strstr(pstr, pat)) != NULL)
    {
        count++;

        /* Increase the cache size when necessary. */
        if (cache_sz < count)
        {
            cache_sz += cache_sz_inc;
            pos_cache = (ptrdiff_t*) realloc(pos_cache, sizeof(*pos_cache) * cache_sz);
            if (pos_cache == NULL)
                goto end_str_repl;
            cache_sz_inc *= cache_sz_inc_factor;
            if (cache_sz_inc > cache_sz_inc_max)
                cache_sz_inc = cache_sz_inc_max;
        }

        pos_cache[count-1] = pstr2 - str;
        pstr = pstr2 + patlen;
    }

    orglen = pstr - str + strlen(pstr);

    /* Allocate memory for the post-replacement string. */
    if (count > 0)
    {
        repllen = strlen(repl);
        retlen = orglen + (repllen - patlen) * count;
    }
    else
        retlen = orglen;
    ret = (char*)malloc(retlen + 1);
    if (ret == NULL)
        goto end_str_repl;

    if (count == 0)
        /* If no matches, then just duplicate the string. */
        strcpy(ret, str);
    else
    {
        /* Otherwise, duplicate the string whilst performing
         * the replacements using the position cache. */
        pret = ret;
        memcpy(pret, str, pos_cache[0]);
        pret += pos_cache[0];
        for (i = 0; i < count; i++)
        {
            memcpy(pret, repl, repllen);
            pret += repllen;
            pstr = str + pos_cache[i] + patlen;
            cpylen = (i == count-1 ? orglen : (size_t)pos_cache[i+1]) - pos_cache[i] - patlen;
            memcpy(pret, pstr, cpylen);
            pret += cpylen;
        }
        ret[retlen] = '\0';
    }

end_str_repl:
    /* Free the cache and return the post-replacement string,
     * which will be NULL in the event of an error. */
    free((void*)pos_cache);
    return ret;
}
