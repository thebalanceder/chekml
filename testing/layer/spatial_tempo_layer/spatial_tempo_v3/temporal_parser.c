#define _GNU_SOURCE
#include "temporal_parser.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static char *trim(char *s) {
    while (*s == ' ' || *s == '\t') ++s;
    char *end = s + strlen(s) - 1;
    while (end >= s && (*end == ' ' || *end == '\t' || *end == '\n')) *end-- = '\0';
    return s;
}

int temporal_load(const char *filename, TemporalFormula *out) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", filename);
        return -1;
    }

    memset(out, 0, sizeof(TemporalFormula));
    out->k_init = 0.1f; // Default
    out->history_length = 5; // Default

    char line[256];
    char *section = NULL;

    while (fgets(line, sizeof(line), f)) {
        char *s = trim(line);
        if (*s == '#' || *s == '\0') continue; // Skip comments and empty lines

        if (s[0] == '[' && s[strlen(s)-1] == ']') {
            free(section);
            s[strlen(s)-1] = '\0';
            section = strdup(s + 1);
            continue;
        }

        if (!section) continue;

        if (strcmp(section, "vars") == 0) {
            if (out->var_count >= MAX_TEMPORAL_VARS) {
                fprintf(stderr, "Too many variables in %s\n", filename);
                free(section); fclose(f); temporal_free(out); return -2;
            }
            out->vars[out->var_count++] = strdup(s);
        } else if (strcmp(section, "formula") == 0) {
            if (out->formula) free(out->formula);
            out->formula = strdup(s);
        } else if (strcmp(section, "params") == 0) {
            char *key = strtok(s, " \t");
            char *value = strtok(NULL, " \t");
            if (key && value) {
                if (strcmp(key, "k_init") == 0) {
                    out->k_init = atof(value);
                } else if (strcmp(key, "history_length") == 0) {
                    out->history_length = atoi(value);
                    if (out->history_length < 1 || out->history_length > 100) {
                        fprintf(stderr, "Invalid history_length in %s\n", filename);
                        free(section); fclose(f); temporal_free(out); return -3;
                    }
                }
            }
        }
    }

    free(section);
    fclose(f);

    if (!out->formula || out->var_count == 0) {
        fprintf(stderr, "Missing formula or variables in %s\n", filename);
        temporal_free(out);
        return -4;
    }

    return 0;
}

void temporal_free(TemporalFormula *f) {
    if (!f) return;
    free(f->formula);
    for (int i = 0; i < f->var_count; ++i) free(f->vars[i]);
    memset(f, 0, sizeof(TemporalFormula));
}
