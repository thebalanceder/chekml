#ifndef TEMPORAL_PARSER_H
#define TEMPORAL_PARSER_H

#include <stdlib.h>

#define MAX_TEMPORAL_VARS 16

typedef struct {
    char *formula;                // Temporal formula (e.g., "w * (1 + k * g)")
    char *vars[MAX_TEMPORAL_VARS]; // Variable names (e.g., "w", "k", "g")
    int var_count;                // Number of variables
    float k_init;                 // Initial value for learnable parameter k
    int history_length;           // Number of epochs to track history
} TemporalFormula;

int temporal_load(const char *filename, TemporalFormula *out);
void temporal_free(TemporalFormula *f);

#endif // TEMPORAL_PARSER_H
