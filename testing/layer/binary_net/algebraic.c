#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "custom_operator.h"

// Apply operator for a single scalar input
void apply_operator_tensor(float w, float x, float *C) {
    float D = 0.0f, E = 0.0f, D_next, E_next;
    custom_operator_diff(w, x, D, E, C, &D_next, &E_next);
}

// Train algebraic method
void train_algebraic(AlgebraicConfig *config, int epochs, float lr) {
    if (config->num_data == 0) {
        printf("Error: No data points available for training\n");
        return;
    }

    float w1 = config->w1;
    float w2 = config->w2;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        float grad_w1 = 0.0f, grad_w2 = 0.0f;

        // Compute loss and gradients for each data point
        for (int d = 0; d < config->num_data; d++) {
            float C1, C2;
            apply_operator_tensor(w1, config->x1[d], &C1);
            apply_operator_tensor(w2, config->x2[d], &C2);
            float sum = C1 + C2;
            float diff = sum - config->y_target[d];
            float loss = diff * diff;
            total_loss += loss;

            // Numerical gradients
            float epsilon = 1e-4;
            float C1_plus, C2_plus;
            apply_operator_tensor(w1 + epsilon, config->x1[d], &C1_plus);
            float loss_plus_w1 = (C1_plus + C2 - config->y_target[d]) * (C1_plus + C2 - config->y_target[d]);
            grad_w1 += isnan(loss_plus_w1) || isinf(loss_plus_w1) ? 0.0f : (loss_plus_w1 - loss) / epsilon;

            apply_operator_tensor(w2 + epsilon, config->x2[d], &C2_plus);
            float loss_plus_w2 = (C1 + C2_plus - config->y_target[d]) * (C1 + C2_plus - config->y_target[d]);
            grad_w2 += isnan(loss_plus_w2) || isinf(loss_plus_w2) ? 0.0f : (loss_plus_w2 - loss) / epsilon;

            // Debug: Print if loss is invalid, only once per problematic epoch
            if (loss < 1e-6 || isnan(loss) || isinf(loss)) {
                if (epoch % 200 == 0 || epoch == 0) {
                    printf("Debug: epoch = %d, data_idx = %d, x1 = %.3f, x2 = %.3f, y_target = %.3f\n", 
                           epoch, d, config->x1[d], config->x2[d], config->y_target[d]);
                    printf("Debug: w1 = %.3f, w2 = %.3f, C1 = %.3f, C2 = %.3f, sum = %.3f, loss = %.6f\n", 
                           w1, w2, C1, C2, sum, loss);
                }
            }
        }
        total_loss /= config->num_data;
        grad_w1 /= config->num_data;
        grad_w2 /= config->num_data;

        // Update weights
        w1 -= lr * grad_w1;
        w2 -= lr * grad_w2;

        // Clamp weights to prevent overflow
        if (w1 > 10.0f) w1 = 10.0f;
        if (w1 < -10.0f) w1 = -10.0f;
        if (w2 > 10.0f) w2 = 10.0f;
        if (w2 < -10.0f) w2 = -10.0f;

        if ((epoch + 1) % 200 == 0 || epoch == 0) {
            printf("Epoch %d  Loss %.6f\n", epoch + 1, total_loss);
        }
    }

    // Store final weights
    config->w1 = w1;
    config->w2 = w2;

    // Print final results
    printf("\nFinal results for %d data points:\n", config->num_data);
    for (int d = 0; d < config->num_data; d++) {
        float C1, C2;
        apply_operator_tensor(w1, config->x1[d], &C1);
        apply_operator_tensor(w2, config->x2[d], &C2);
        float sum = C1 + C2;
        printf("Data point %d: x1=%.3f, x2=%.3f, y_target=%.3f, sum=%.3f\n",
               d, config->x1[d], config->x2[d], config->y_target[d], sum);
    }
    printf("Final w1: %.3f\n", w1);
    printf("Final w2: %.3f\n", w2);
}
