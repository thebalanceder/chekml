#ifndef TEMPORAL_PARSER_H
#define TEMPORAL_PARSER_H

#include <stdlib.h>

#define MAX_TEMPORAL_VARS 16
#define MAX_LEARNABLE_PARAMS 8

typedef struct {
    char *formula;                // Temporal formula (e.g., "w * (1 + k1 * g + k2 * g^2)")
    char *vars[MAX_TEMPORAL_VARS]; // Variable names (e.g., "w", "k1", "k2", "g")
    int var_count;                // Number of variables
    char *param_names[MAX_LEARNABLE_PARAMS]; // Names of learnable parameters (e.g., "k1", "k2")
    float param_inits[MAX_LEARNABLE_PARAMS]; // Initial values for learnable parameters
    int param_count;              // Number of learnable parameters
    int history_length;           // Number of epochs to track history
    char *testing_formula;        // Custom testing formula (e.g., "g > 0.5")
} TemporalFormula;

int temporal_load(const char *filename, TemporalFormula *out);
void temporal_free(TemporalFormula *f);

#endif // TEMPORAL_PARSER_H
