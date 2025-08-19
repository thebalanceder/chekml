#ifndef CUSTOM_OPERATOR_H
#define CUSTOM_OPERATOR_H

#define MAX_DATA_POINTS 100 // Max number of data points

// Structure for algebraic method inputs
typedef struct {
    float *x1, *x2, *y_target;  // Arrays of scalar inputs and targets
    int num_data;              // Number of data points
    float w1, w2;              // Trainable scalar weights
} AlgebraicConfig;

// Structure for STE method inputs
typedef struct {
    int *x1, *x2;              // Arrays of input integers
    int *y_target;             // Array of target integers
    int num_data;              // Number of data points
    int bits;                  // Number of bits
    float *w1_logits, *w2_logits;  // Trainable logits
} STEConfig;

// Training configuration
typedef struct {
    char method[20];           // "algebraic" or "ste"
    int epochs;
    float lr;
    AlgebraicConfig alg_config;
    STEConfig ste_config;
} Config;

// Binary custom operator
void custom_operator_bits(int A, int B, int D_prev, int E_prev, int *C, int *D_next, int *E_next);

// Differentiable custom operator
void custom_operator_diff(float A, float B, float D_prev, float E_prev, float *C, float *D_next, float *E_next);

// Algebraic method
void apply_operator_tensor(float w, float x, float *C);
void train_algebraic(AlgebraicConfig *config, int epochs, float lr);

// STE method
float sigmoid(float x);
void sigmoid_to_ste_bits(float *logits, int bits, float *bits_ste, int *hard_bits);
int apply_operator_discrete_bits(int *w_bits, int *x_bits, int bits);
void train_ste(STEConfig *config, int epochs, float lr);

// .sybl parser
int parse_sybl(const char *filename, Config *config);
void free_config(Config *config);

#endif
