#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "custom_operator.h"

// Sigmoid function
float sigmoid(float x) {
    if (x > 20.0f) return 1.0f;  // Clamp to avoid overflow
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

// STE: Convert logits to bits
void sigmoid_to_ste_bits(float *logits, int bits, float *bits_ste, int *hard_bits) {
    for (int i = 0; i < bits; i++) {
        float prob = sigmoid(logits[i]);
        hard_bits[i] = prob > 0.5 ? 1 : 0;
        bits_ste[i] = prob;
    }
}

// Apply discrete operator bit-by-bit
int apply_operator_discrete_bits(int *w_bits, int *x_bits, int bits) {
    int result = 0, D = 0, E = 0;
    for (int i = 0; i < bits; i++) {
        int C, D_next, E_next;
        custom_operator_bits(w_bits[i], x_bits[i], D, E, &C, &D_next, &E_next);
        result |= (C << i);
        D = D_next;
        E = E_next;
    }
    return result;
}

// Compute soft surrogate and gradients for one data point
float compute_soft_loss_and_grads_single(STEConfig *config, int data_idx, float *soft_sum, int *discrete_sum, float *grad_w1_logits, float *grad_w2_logits) {
    int bits = config->bits;
    float *w1_probs = (float *)calloc(bits, sizeof(float));
    float *w2_probs = (float *)calloc(bits, sizeof(float));
    int *w1_bits = (int *)calloc(bits, sizeof(int));
    int *w2_bits = (int *)calloc(bits, sizeof(int));
    int *x1_bits = (int *)calloc(bits, sizeof(int));
    int *x2_bits = (int *)calloc(bits, sizeof(int));
    float *soft_term1_bits = (float *)calloc(bits, sizeof(float));
    float *soft_term2_bits = (float *)calloc(bits, sizeof(float));

    // Convert inputs to bits and compute probabilities
    for (int i = 0; i < bits; i++) {
        x1_bits[i] = (config->x1[data_idx] >> i) & 1;
        x2_bits[i] = (config->x2[data_idx] >> i) & 1;
        w1_probs[i] = sigmoid(config->w1_logits[i]);
        w2_probs[i] = sigmoid(config->w2_logits[i]);
        w1_bits[i] = w1_probs[i] > 0.5 ? 1 : 0;
        w2_bits[i] = w2_probs[i] > 0.5 ? 1 : 0;
    }

    // Discrete forward
    int term1_int = apply_operator_discrete_bits(w1_bits, x1_bits, bits);
    int term2_int = apply_operator_discrete_bits(w2_bits, x2_bits, bits);
    *discrete_sum = term1_int + term2_int;

    // Soft surrogate
    float D = 0.0f, E = 0.0f, D_next, E_next;
    for (int i = 0; i < bits; i++) {
        float C;
        custom_operator_diff(w1_probs[i], (float)x1_bits[i], D, E, &C, &D_next, &E_next);
        soft_term1_bits[i] = C;
        D = D_next;
        E = E_next;
    }
    D = 0.0f; E = 0.0f;
    for (int i = 0; i < bits; i++) {
        float C;
        custom_operator_diff(w2_probs[i], (float)x2_bits[i], D, E, &C, &D_next, &E_next);
        soft_term2_bits[i] = C;
        D = D_next;
        E = E_next;
    }

    float soft_term1_val = 0.0f, soft_term2_val = 0.0f;
    for (int i = 0; i < bits; i++) {
        soft_term1_val += (1 << i) * soft_term1_bits[i];
        soft_term2_val += (1 << i) * soft_term2_bits[i];
    }
    *soft_sum = soft_term1_val + soft_term2_val;

    // Loss calculation
    float diff = *soft_sum - (float)config->y_target[data_idx];
    float loss = diff * diff;

    // Numerical gradients for logits
    float epsilon = 1e-4;
    for (int i = 0; i < bits; i++) {
        // Gradient for w1_logits[i]
        float orig_logit = config->w1_logits[i];
        config->w1_logits[i] += epsilon;
        w1_probs[i] = sigmoid(config->w1_logits[i]);
        D = 0.0f; E = 0.0f;
        for (int j = 0; j < bits; j++) {
            float C;
            custom_operator_diff(w1_probs[j], (float)x1_bits[j], D, E, &C, &D_next, &E_next);
            soft_term1_bits[j] = C;
            D = D_next;
            E = E_next;
        }
        float soft_term1_plus = 0.0f;
        for (int j = 0; j < bits; j++) {
            soft_term1_plus += (1 << j) * soft_term1_bits[j];
        }
        float loss_plus = (soft_term1_plus + soft_term2_val - (float)config->y_target[data_idx]) * 
                          (soft_term1_plus + soft_term2_val - (float)config->y_target[data_idx]);
        grad_w1_logits[i] = isnan(loss_plus) || isinf(loss_plus) ? 0.0f : (loss_plus - loss) / epsilon;
        config->w1_logits[i] = orig_logit;
        w1_probs[i] = sigmoid(orig_logit);

        // Gradient for w2_logits[i]
        orig_logit = config->w2_logits[i];
        config->w2_logits[i] += epsilon;
        w2_probs[i] = sigmoid(config->w2_logits[i]);
        D = 0.0f; E = 0.0f;
        for (int j = 0; j < bits; j++) {
            float C;
            custom_operator_diff(w2_probs[j], (float)x2_bits[j], D, E, &C, &D_next, &E_next);
            soft_term2_bits[j] = C;
            D = D_next;
            E = E_next;
        }
        float soft_term2_plus = 0.0f;
        for (int j = 0; j < bits; j++) {
            soft_term2_plus += (1 << j) * soft_term2_bits[j];
        }
        loss_plus = (soft_term1_val + soft_term2_plus - (float)config->y_target[data_idx]) * 
                    (soft_term1_val + soft_term2_plus - (float)config->y_target[data_idx]);
        grad_w2_logits[i] = isnan(loss_plus) || isinf(loss_plus) ? 0.0f : (loss_plus - loss) / epsilon;
        config->w2_logits[i] = orig_logit;
        w2_probs[i] = sigmoid(orig_logit);
    }

    // Debug: Print intermediate values if loss is invalid
    if (loss < 1e-6 || isnan(loss) || isinf(loss)) {
        printf("Debug: data_idx = %d, x1 = %d, x2 = %d, y_target = %d\n", 
               data_idx, config->x1[data_idx], config->x2[data_idx], config->y_target[data_idx]);
        printf("Debug: w1_probs = [");
        for (int i = 0; i < bits; i++) printf("%.3f%s", w1_probs[i], i < bits - 1 ? ", " : "]\n");
        printf("Debug: w2_probs = [");
        for (int i = 0; i < bits; i++) printf("%.3f%s", w2_probs[i], i < bits - 1 ? ", " : "]\n");
        printf("Debug: soft_term1_bits = [");
        for (int i = 0; i < bits; i++) printf("%.3f%s", soft_term1_bits[i], i < bits - 1 ? ", " : "]\n");
        printf("Debug: soft_term2_bits = [");
        for (int i = 0; i < bits; i++) printf("%.3f%s", soft_term2_bits[i], i < bits - 1 ? ", " : "]\n");
        printf("Debug: soft_term1_val = %.3f, soft_term2_val = %.3f, soft_sum = %.3f\n", 
               soft_term1_val, soft_term2_val, *soft_sum);
    }

    free(w1_probs);
    free(w2_probs);
    free(w1_bits);
    free(w2_bits);
    free(x1_bits);
    free(x2_bits);
    free(soft_term1_bits);
    free(soft_term2_bits);
    return loss;
}

// Train STE method
void train_ste(STEConfig *config, int epochs, float lr) {
    if (config->num_data == 0) {
        printf("Error: No data points available for training\n");
        return;
    }

    float *grad_w1_logits = (float *)calloc(config->bits, sizeof(float));
    float *grad_w2_logits = (float *)calloc(config->bits, sizeof(float));

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        float total_soft_sum = 0.0f;
        int total_discrete_sum = 0;

        // Zero gradients
        for (int i = 0; i < config->bits; i++) {
            grad_w1_logits[i] = 0.0f;
            grad_w2_logits[i] = 0.0f;
        }

        // Compute loss and gradients for each data point
        for (int d = 0; d < config->num_data; d++) {
            float soft_sum;
            int discrete_sum;
            float *temp_grad_w1 = (float *)calloc(config->bits, sizeof(float));
            float *temp_grad_w2 = (float *)calloc(config->bits, sizeof(float));

            float loss = compute_soft_loss_and_grads_single(config, d, &soft_sum, &discrete_sum, temp_grad_w1, temp_grad_w2);
            total_loss += loss;
            total_soft_sum += soft_sum;
            total_discrete_sum += discrete_sum;

            // Accumulate gradients
            for (int i = 0; i < config->bits; i++) {
                grad_w1_logits[i] += isnan(temp_grad_w1[i]) || isinf(temp_grad_w1[i]) ? 0.0f : temp_grad_w1[i] / config->num_data;
                grad_w2_logits[i] += isnan(temp_grad_w2[i]) || isinf(temp_grad_w2[i]) ? 0.0f : temp_grad_w2[i] / config->num_data;
            }

            free(temp_grad_w1);
            free(temp_grad_w2);
        }
        total_loss /= config->num_data;
        total_soft_sum /= config->num_data;
        total_discrete_sum /= config->num_data;

        // Update logits using gradients
        for (int i = 0; i < config->bits; i++) {
            config->w1_logits[i] -= lr * grad_w1_logits[i];
            config->w2_logits[i] -= lr * grad_w2_logits[i];
            // Clamp logits to prevent overflow
            if (config->w1_logits[i] > 20.0f) config->w1_logits[i] = 20.0f;
            if (config->w1_logits[i] < -20.0f) config->w1_logits[i] = -20.0f;
            if (config->w2_logits[i] > 20.0f) config->w2_logits[i] = 20.0f;
            if (config->w2_logits[i] < -20.0f) config->w2_logits[i] = -20.0f;
        }

        if ((epoch + 1) % 200 == 0 || epoch == 0) {
            printf("Epoch %d  Loss %.6f  (soft=%.3f, discrete=%d)\n", 
                   epoch + 1, total_loss, total_soft_sum, total_discrete_sum);
        }
    }

    // Print final results for each data point
    int *w1_bits = (int *)calloc(config->bits, sizeof(int));
    int *w2_bits = (int *)calloc(config->bits, sizeof(int));
    for (int i = 0; i < config->bits; i++) {
        w1_bits[i] = sigmoid(config->w1_logits[i]) > 0.5 ? 1 : 0;
        w2_bits[i] = sigmoid(config->w2_logits[i]) > 0.5 ? 1 : 0;
    }

    printf("\nFinal results for %d data points:\n", config->num_data);
    for (int d = 0; d < config->num_data; d++) {
        int *x1_bits = (int *)calloc(config->bits, sizeof(int));
        int *x2_bits = (int *)calloc(config->bits, sizeof(int));
        for (int i = 0; i < config->bits; i++) {
            x1_bits[i] = (config->x1[d] >> i) & 1;
            x2_bits[i] = (config->x2[d] >> i) & 1;
        }
        int final_discrete_sum = apply_operator_discrete_bits(w1_bits, x1_bits, config->bits) +
                                 apply_operator_discrete_bits(w2_bits, x2_bits, config->bits);
        float final_soft_sum = 0.0f;
        float D = 0.0f, E = 0.0f, D_next, E_next;
        float *soft_term1_bits = (float *)calloc(config->bits, sizeof(float));
        float *soft_term2_bits = (float *)calloc(config->bits, sizeof(float));
        for (int i = 0; i < config->bits; i++) {
            float C;
            custom_operator_diff(sigmoid(config->w1_logits[i]), (float)x1_bits[i], D, E, &C, &D_next, &E_next);
            soft_term1_bits[i] = C;
            D = D_next;
            E = E_next;
        }
        D = 0.0f; E = 0.0f;
        for (int i = 0; i < config->bits; i++) {
            float C;
            custom_operator_diff(sigmoid(config->w2_logits[i]), (float)x2_bits[i], D, E, &C, &D_next, &E_next);
            soft_term2_bits[i] = C;
            D = D_next;
            E = E_next;
        }
        for (int i = 0; i < config->bits; i++) {
            final_soft_sum += (1 << i) * (soft_term1_bits[i] + soft_term2_bits[i]);
        }
        printf("Data point %d: x1=%d, x2=%d, y_target=%d, discrete_sum=%d, soft_sum=%.3f\n",
               d, config->x1[d], config->x2[d], config->y_target[d], final_discrete_sum, final_soft_sum);
        free(x1_bits);
        free(x2_bits);
        free(soft_term1_bits);
        free(soft_term2_bits);
    }

    printf("Final w1 bits (LSB->MSB): [");
    for (int i = 0; i < config->bits; i++) {
        printf("%d%s", w1_bits[i], i < config->bits - 1 ? ", " : "]\n");
    }
    printf("Final w2 bits (LSB->MSB): [");
    for (int i = 0; i < config->bits; i++) {
        printf("%d%s", w2_bits[i], i < config->bits - 1 ? ", " : "]\n");
    }

    free(w1_bits);
    free(w2_bits);
    free(grad_w1_logits);
    free(grad_w2_logits);
}
