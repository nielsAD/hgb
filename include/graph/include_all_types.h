// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "util/ubiquitous.h"
#include "util/string.h"

#ifndef GRAPH_INCLUDE_FILE
    #error Graph template file not declared!
#else
    #define _GRAPH_NAME(SUF,ID) STATIC_CONCAT(GRAPH_BASENAME,SUF,ID)

    #if defined(GRAPH_V_TYPE) || defined(GRAPH_E_TYPE) || defined(GRAPH_NAME)
        #error Graph template macros already defined!
    #endif

    #define GRAPH_NAME(ID) _GRAPH_NAME(_v,ID)
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_p,ID)
    #define GRAPH_V_TYPE void*
    #define GRAPH_E_TYPE void*
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_b,ID)
    #define GRAPH_V_TYPE bool
    #define GRAPH_E_TYPE bool
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_f,ID)
    #define GRAPH_V_TYPE float
    #define GRAPH_E_TYPE float
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_d,ID)
    #define GRAPH_V_TYPE double
    #define GRAPH_E_TYPE double
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_ld,ID)
    #define GRAPH_V_TYPE long double
    #define GRAPH_E_TYPE long double
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_c,ID)
    #define GRAPH_V_TYPE char
    #define GRAPH_E_TYPE char
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_s,ID)
    #define GRAPH_V_TYPE short int
    #define GRAPH_E_TYPE short int
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_i,ID)
    #define GRAPH_V_TYPE int
    #define GRAPH_E_TYPE int
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_li,ID)
    #define GRAPH_V_TYPE long int
    #define GRAPH_E_TYPE long int
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_lli,ID)
    #define GRAPH_V_TYPE long long int
    #define GRAPH_E_TYPE long long int
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_uc,ID)
    #define GRAPH_V_TYPE unsigned char
    #define GRAPH_E_TYPE unsigned char
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_us,ID)
    #define GRAPH_V_TYPE unsigned short int
    #define GRAPH_E_TYPE unsigned short int
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_u,ID)
    #define GRAPH_V_TYPE unsigned int
    #define GRAPH_E_TYPE unsigned int
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_lu,ID)
    #define GRAPH_V_TYPE unsigned long int
    #define GRAPH_E_TYPE unsigned long int
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_llu,ID)
    #define GRAPH_V_TYPE unsigned long long int
    #define GRAPH_E_TYPE unsigned long long int
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME

    #define GRAPH_NAME(ID) _GRAPH_NAME(_sz,ID)
    #define GRAPH_V_TYPE size_t
    #define GRAPH_E_TYPE size_t
    #include GRAPH_INCLUDE_FILE
    #undef GRAPH_E_TYPE
    #undef GRAPH_V_TYPE
    #undef GRAPH_NAME
    #undef _GRAPH_NAME
#endif
