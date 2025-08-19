#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "custom_operator.h"

// Parse a data triple from a string like "0.2,0.1,0.3" or "[0.2,0.1,0.3]"
int parse_data_triple(const char *str, float *x1, float *x2, float *y_target, int is_ste, int line_num) {
    char *copy = strdup(str);
    char *start = copy;
    while (isspace(*start)) start++;
    if (*start == '[') start++;
    char *end = start + strlen(start) - 1;
    while (end >= start && (isspace(*end) || *end == ']')) *end-- = '\0';

    char *token = strtok(start, ",");
    if (!token || *token == '\0') {
        printf("Error: Empty or invalid x1 in triple at line %d: %s\n", line_num, str);
        free(copy);
        return -1;
    }
    while (isspace(*token)) token++;
    if (!isdigit(*token) && *token != '-' && *token != '.') {
        printf("Error: Invalid x1 in triple at line %d: %s\n", line_num, str);
        free(copy);
        return -1;
    }
    *x1 = is_ste ? atoi(token) : atof(token);
    token = strtok(NULL, ",");
    if (!token || *token == '\0') {
        printf("Error: Missing or invalid x2 in triple at line %d: %s\n", line_num, str);
        free(copy);
        return -1;
    }
    while (isspace(*token)) token++;
    if (!isdigit(*token) && *token != '-' && *token != '.') {
        printf("Error: Invalid x2 in triple at line %d: %s\n", line_num, str);
        free(copy);
        return -1;
    }
    *x2 = is_ste ? atoi(token) : atof(token);
    token = strtok(NULL, ",");
    if (!token || *token == '\0') {
        printf("Error: Missing or invalid y_target in triple at line %d: %s\n", line_num, str);
        free(copy);
        return -1;
    }
    while (isspace(*token)) token++;
    if (!isdigit(*token) && *token != '-' && *token != '.') {
        printf("Error: Invalid y_target in triple at line %d: %s\n", line_num, str);
        free(copy);
        return -1;
    }
    *y_target = is_ste ? atoi(token) : atof(token);
    token = strtok(NULL, ",");
    if (token && *token != '\0') {
        printf("Error: Extra tokens in triple at line %d: %s\n", line_num, str);
        free(copy);
        return -1;
    }
    free(copy);
    return 0;
}

int parse_sybl(const char *filename, Config *config) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return -1;
    }

    char line[256];
    int line_num = 0;
    int data_count = 0;
    char *data_buffer = NULL;
    int data_buffer_size = 0;
    int is_ste = 0;

    config->alg_config.x1 = (float *)calloc(MAX_DATA_POINTS, sizeof(float));
    config->alg_config.x2 = (float *)calloc(MAX_DATA_POINTS, sizeof(float));
    config->alg_config.y_target = (float *)calloc(MAX_DATA_POINTS, sizeof(float));
    config->alg_config.num_data = 0;
    config->ste_config.x1 = (int *)calloc(MAX_DATA_POINTS, sizeof(int));
    config->ste_config.x2 = (int *)calloc(MAX_DATA_POINTS, sizeof(int));
    config->ste_config.y_target = (int *)calloc(MAX_DATA_POINTS, sizeof(int));
    config->ste_config.num_data = 0;

    while (fgets(line, sizeof(line), file)) {
        line_num++;
        line[strcspn(line, "\n\r")] = 0; // Remove \n and \r
        char *key = strtok(line, "=");
        char *value = strtok(NULL, "");
        if (!key || !value) continue;

        while (isspace(*key)) key++;
        while (isspace(*value)) value++;

        if (strcmp(key, "method") == 0) {
            strncpy(config->method, value, sizeof(config->method) - 1);
            is_ste = strcmp(config->method, "ste") == 0;
        } else if (strcmp(key, "x1") == 0 && strchr(value, '[') == NULL) {
            if (is_ste) {
                config->ste_config.x1[0] = atoi(value);
                config->ste_config.num_data = 1;
            } else {
                config->alg_config.x1[0] = atof(value);
                config->alg_config.num_data = 1;
            }
        } else if (strcmp(key, "x2") == 0 && strchr(value, '[') == NULL) {
            if (is_ste) {
                config->ste_config.x2[0] = atoi(value);
            } else {
                config->alg_config.x2[0] = atof(value);
            }
        } else if (strcmp(key, "y_target") == 0 && strchr(value, '[') == NULL) {
            if (is_ste) {
                config->ste_config.y_target[0] = atoi(value);
            } else {
                config->alg_config.y_target[0] = atof(value);
            }
        } else if (strcmp(key, "data") == 0) {
            data_buffer = realloc(data_buffer, data_buffer_size + strlen(value) + 1);
            strcpy(data_buffer + data_buffer_size, value);
            data_buffer_size += strlen(value);
            int data_start_line = line_num;
            while (fgets(line, sizeof(line), file)) {
                line_num++;
                line[strcspn(line, "\n\r")] = 0;
                if (strchr(line, ']') && strlen(line) < 3) break;
                data_buffer = realloc(data_buffer, data_buffer_size + strlen(line) + 2);
                strcat(data_buffer + data_buffer_size, " ");
                strcat(data_buffer + data_buffer_size + 1, line);
                data_buffer_size += strlen(line) + 1;
            }
            printf("Debug: Raw data buffer: %s\n", data_buffer);

            // Custom parsing to extract full triples
            char *start = data_buffer;
            while (*start && *start != '[') start++;
            if (*start != '[') {
                printf("Error: Missing opening bracket in data field at line %d\n", data_start_line);
                free(data_buffer);
                fclose(file);
                return -1;
            }
            start++;
            char *end = strchr(start, ']');
            if (!end) {
                printf("Error: Missing closing bracket in data field at line %d\n", data_start_line);
                free(data_buffer);
                fclose(file);
                return -1;
            }

            // Parse triples between [ and ]
            char *triple_start = start;
            int triple_line = data_start_line + 1;
            while (triple_start < end && data_count < MAX_DATA_POINTS) {
                // Find end of current triple
                char *triple_end = triple_start;
                int bracket_count = 0;
                while (triple_end < end && (*triple_end != ']' || bracket_count > 0)) {
                    if (*triple_end == '[') bracket_count++;
                    if (*triple_end == ']') bracket_count--;
                    triple_end++;
                }
                if (*triple_end == ']') triple_end--;
                while (triple_end >= triple_start && isspace(*triple_end)) triple_end--;
                if (triple_end < triple_start) {
                    triple_start = triple_end + 1;
                    continue;
                }
                *(triple_end + 1) = '\0';

                // Skip leading whitespace or commas
                while (triple_start < triple_end && (isspace(*triple_start) || *triple_start == ',')) {
                    triple_start++;
                    if (*triple_start == '\n' || *triple_start == '\r') triple_line++;
                }
                if (*triple_start == '\0') {
                    triple_start = triple_end + 2;
                    continue;
                }

                printf("Parsing data triple: %s\n", triple_start);
                if (is_ste) {
                    float x1, x2, y_target;
                    if (parse_data_triple(triple_start, &x1, &x2, &y_target, 1, triple_line) == 0) {
                        config->ste_config.x1[data_count] = (int)x1;
                        config->ste_config.x2[data_count] = (int)x2;
                        config->ste_config.y_target[data_count] = (int)y_target;
                        printf("Parsed data point %d: x1=%d, x2=%d, y_target=%d\n", 
                               data_count, config->ste_config.x1[data_count], 
                               config->ste_config.x2[data_count], config->ste_config.y_target[data_count]);
                        data_count++;
                    } else {
                        printf("Error: Invalid data triple at line %d: %s\n", triple_line, triple_start);
                    }
                } else {
                    if (parse_data_triple(triple_start, &config->alg_config.x1[data_count], 
                                          &config->alg_config.x2[data_count], 
                                          &config->alg_config.y_target[data_count], 0, triple_line) == 0) {
                        printf("Parsed data point %d: x1=%.3f, x2=%.3f, y_target=%.3f\n", 
                               data_count, config->alg_config.x1[data_count], 
                               config->alg_config.x2[data_count], config->alg_config.y_target[data_count]);
                        data_count++;
                    } else {
                        printf("Error: Invalid data triple at line %d: %s\n", triple_line, triple_start);
                    }
                }
                triple_start = triple_end + 2;
                triple_line++;
            }
            if (is_ste) {
                config->ste_config.num_data = data_count;
            } else {
                config->alg_config.num_data = data_count;
            }
            free(data_buffer);
            data_buffer = NULL;
            data_buffer_size = 0;
        } else if (strcmp(key, "bits") == 0) {
            config->ste_config.bits = atoi(value);
        } else if (strcmp(key, "epochs") == 0) {
            config->epochs = atoi(value);
        } else if (strcmp(key, "lr") == 0) {
            config->lr = atof(value);
        }
    }
    fclose(file);

    if (strcmp(config->method, "ste") == 0 && config->ste_config.num_data == 0) {
        printf("Error: No valid data points parsed for STE\n");
        free(config->ste_config.x1);
        free(config->ste_config.x2);
        free(config->ste_config.y_target);
        return -1;
    } else if (strcmp(config->method, "algebraic") == 0 && config->alg_config.num_data == 0) {
        printf("Error: No valid data points parsed for algebraic\n");
        free(config->alg_config.x1);
        free(config->alg_config.x2);
        free(config->alg_config.y_target);
        return -1;
    }

    // Initialize weights
    if (strcmp(config->method, "algebraic") == 0) {
        config->alg_config.w1 = (float)rand() / RAND_MAX;
        config->alg_config.w2 = (float)rand() / RAND_MAX;
    } else if (strcmp(config->method, "ste") == 0) {
        config->ste_config.w1_logits = (float *)calloc(config->ste_config.bits, sizeof(float));
        config->ste_config.w2_logits = (float *)calloc(config->ste_config.bits, sizeof(float));
        for (int i = 0; i < config->ste_config.bits; i++) {
            config->ste_config.w1_logits[i] = ((float)rand() / RAND_MAX - 0.5) * 5.0;
            config->ste_config.w2_logits[i] = ((float)rand() / RAND_MAX - 0.5) * 5.0;
        }
    }
    return 0;
}

void free_config(Config *config) {
    if (strcmp(config->method, "algebraic") == 0) {
        free(config->alg_config.x1);
        free(config->alg_config.x2);
        free(config->alg_config.y_target);
    } else if (strcmp(config->method, "ste") == 0) {
        free(config->ste_config.x1);
        free(config->ste_config.x2);
        free(config->ste_config.y_target);
        free(config->ste_config.w1_logits);
        free(config->ste_config.w2_logits);
    }
}
