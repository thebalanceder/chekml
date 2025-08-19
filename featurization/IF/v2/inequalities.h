#ifndef INEQUALITIES_H
#define INEQUALITIES_H

// Structure declarations
typedef struct {
    const char* name;
    double (*func)(double*, int);
} Inequality;

typedef struct {
    const char* name;
    double value;
} Result;

// Function declarations
void compute_features(double* data, int rows, int* cols, int num_cols, int level, 
                     int stage, double* output, int* output_cols, char** output_names);
extern const int num_inequalities;
extern Inequality inequalities[];

#endif
