// kof_parser.c
#define _GNU_SOURCE
#include "kof_parser.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

static char *trim(char *s){
    while(isspace((unsigned char)*s)) s++;
    if(*s==0) return s;
    char *e = s + strlen(s) - 1;
    while(e > s && isspace((unsigned char)*e)) *e-- = 0;
    return s;
}

static int starts_with(const char *s, const char *pfx){
    return strncmp(s, pfx, strlen(pfx)) == 0;
}

void kof_free(KOFKernel *k){
    if(!k) return;
    free(k->anchor);
    free(k->offsets);
    free(k->weights);
    free(k->operation_formula);
    free(k->input_encode_formula);
    free(k->kernel_encode_formula);

    if(k->tap_op_formulas){
        for(int i = 0; i < k->k_elems; ++i) free(k->tap_op_formulas[i]);
        free(k->tap_op_formulas);
    }
    if(k->tap_input_formulas){
        for(int i = 0; i < k->k_elems; ++i) free(k->tap_input_formulas[i]);
        free(k->tap_input_formulas);
    }
    if(k->tap_kernel_formulas){
        for(int i = 0; i < k->k_elems; ++i) free(k->tap_kernel_formulas[i]);
        free(k->tap_kernel_formulas);
    }

    memset(k, 0, sizeof(*k));
}

int kof_load(const char *path, KOFKernel *out){
    memset(out, 0, sizeof(*out));
    out->norm = KOF_NORM_NONE;
    out->scale = 1.0f;

    FILE *f = fopen(path, "r");
    if(!f) return -1;

    char line[1024];
    int have_ndim=0, reading_taps=0;
    int parsed_taps=0;
    int *tmp_offsets = NULL;
    float *tmp_weights = NULL;
    char **tap_op_formulas = NULL, **tap_in_formulas = NULL, **tap_k_formulas = NULL;

    while(fgets(line, sizeof(line), f)){
        char *s = trim(line);
        if(*s=='#' || *s=='\0') continue;

        if(starts_with(s, "ndim:")){
            s = trim(s+5);
            out->ndim = atoi(s);
            have_ndim = 1;
            continue;
        }
        if(starts_with(s, "taps:")){
            reading_taps = 1;
            continue;
        }
        if(reading_taps && (isdigit(s[0]) || (s[0]=='-' && isdigit(s[1])))){
            if(!have_ndim){ fclose(f); kof_free(out); return -2; }
            if(!tmp_offsets){
                tmp_offsets = (int*)malloc(sizeof(int) * 1024 * out->ndim);
                tmp_weights = (float*)malloc(sizeof(float) * 1024);
                tap_op_formulas = (char**)calloc(1024, sizeof(char*));
                tap_in_formulas = (char**)calloc(1024, sizeof(char*));
                tap_k_formulas  = (char**)calloc(1024, sizeof(char*));
            }
            if(parsed_taps >= 1024){ fclose(f); kof_free(out); return -3; }

            for(int d=0; d<out->ndim; ++d){
                char *endptr;
                int val = (int)strtol(s, &endptr, 10);
                tmp_offsets[parsed_taps*out->ndim + d] = val;
                s = (endptr && *endptr)? trim(endptr) : endptr;
            }
            tmp_weights[parsed_taps] = (float)atof(s);
            ++parsed_taps;
            continue;
        }

        if(starts_with(s, "operation:")){
            s = trim(s+10);
            if(reading_taps && parsed_taps > 0)
                tap_op_formulas[parsed_taps-1] = strdup(s);
            else
                out->operation_formula = strdup(s);
            continue;
        }
        if(starts_with(s, "input_encode:")){
            s = trim(s+13);
            if(reading_taps && parsed_taps > 0)
                tap_in_formulas[parsed_taps-1] = strdup(s);
            else
                out->input_encode_formula = strdup(s);
            continue;
        }
        if(starts_with(s, "kernel_encode:")){
            s = trim(s+14);
            if(reading_taps && parsed_taps > 0)
                tap_k_formulas[parsed_taps-1] = strdup(s);
            else
                out->kernel_encode_formula = strdup(s);
            continue;
        }
    }
    fclose(f);

    out->k_elems = parsed_taps;
    if(!have_ndim || parsed_taps == 0){ kof_free(out); return -4; }

    out->offsets = (int*)malloc(sizeof(int) * parsed_taps * out->ndim);
    out->weights = (float*)malloc(sizeof(float) * parsed_taps);
    memcpy(out->offsets, tmp_offsets, sizeof(int) * parsed_taps * out->ndim);
    memcpy(out->weights, tmp_weights, sizeof(float) * parsed_taps);

    out->tap_op_formulas     = tap_op_formulas;
    out->tap_input_formulas  = tap_in_formulas;
    out->tap_kernel_formulas = tap_k_formulas;

    free(tmp_offsets);
    free(tmp_weights);
    return 0;
}

