#define _GNU_SOURCE
#include "temporal_parser.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

static char *trim(char *s) {
    while (*s == ' ' || *s == '\t') ++s;
    char *end = s + strlen(s) - 1;
    while (end >= s && (*end == ' ' || *end == '\t' || *end == '\n')) *end-- = '\0';
    return s;
}

static int is_variable_start(char c) {
    return isalpha(c) || c == '_';
}

static int is_variable_char(char c) {
    return isalnum(c) || c == '_';
}

// Skip numerical literals (e.g., "1", "0.5", "-3.14")
static char *skip_number(char *p) {
    if (*p == '-' || *p == '+') p++;
    while (isdigit(*p)) p++;
    if (*p == '.') {
        p++;
        while (isdigit(*p)) p++;
    }
    if (*p == 'e' || *p == 'E') {
        p++;
        if (*p == '-' || *p == '+') p++;
        while (isdigit(*p)) p++;
    }
    return p;
}

// Extract variables from a formula, ignoring numerical constants
static int extract_variables(const char *formula, char **vars, int *var_count, int max_vars) {
    *var_count = 0;
    char *temp = strdup(formula);
    if (!temp) {
        fprintf(stderr, "Memory allocation failed for formula parsing\n");
        return -1;
    }

    char *p = temp;
    while (*p) {
        // Skip whitespace, operators, parentheses, and numbers
        while (*p && (isspace(*p) || strchr("+-*/^()[]{},", *p))) p++;
        if (*p && isdigit(*p)) {
            p = skip_number(p);
            continue;
        }
        if (!*p) break;

        // Check for variable start
        if (is_variable_start(*p)) {
            char *start = p;
            while (*p && is_variable_char(*p)) p++;
            if (p > start) {
                char saved = *p;
                *p = '\0';
                int found = 0;
                for (int i = 0; i < *var_count; ++i) {
                    if (strcmp(vars[i], start) == 0) {
                        found = 1;
                        break;
                    }
                }
                if (!found && *var_count < max_vars) {
                    vars[*var_count] = strdup(start);
                    if (!vars[*var_count]) {
                        for (int i = 0; i < *var_count; ++i) free(vars[i]);
                        free(temp);
                        return -1;
                    }
                    (*var_count)++;
                }
                *p = saved;
            }
        } else {
            p++;
        }
    }
    free(temp);
    return 0;
}

// Validate that all formula variables are in vars list
static int validate_variables(const char *formula, const char **vars, int var_count, const char *filename, const char *context) {
    char *temp_vars[MAX_TEMPORAL_VARS] = {0};
    int temp_var_count = 0;
    if (extract_variables(formula, temp_vars, &temp_var_count, MAX_TEMPORAL_VARS) != 0) {
        for (int i = 0; i < temp_var_count; ++i) free(temp_vars[i]);
        fprintf(stderr, "Failed to extract variables from %s in %s\n", context, filename);
        return -1;
    }

    for (int i = 0; i < temp_var_count; ++i) {
        int found = 0;
        for (int j = 0; j < var_count; ++j) {
            if (strcmp(temp_vars[i], vars[j]) == 0) {
                found = 1;
                break;
            }
        }
        // Allow 'm' for memory formula
        if (!found && strcmp(temp_vars[i], "m") != 0) {
            fprintf(stderr, "Variable '%s' in %s not found in [vars] section of %s\n", temp_vars[i], context, filename);
            for (int k = 0; k < temp_var_count; ++k) free(temp_vars[k]);
            return -1;
        }
    }
    for (int i = 0; i < temp_var_count; ++i) free(temp_vars[i]);
    return 0;
}

// Parse boolean testing formula (e.g., "g > 0.5" -> lhs="g", op=REL_OP_GT, rhs="0.5")
static int parse_boolean_formula(const char *input, BooleanFormula *out, const char *filename) {
    out->lhs = NULL;
    out->rhs = NULL;
    out->op = REL_OP_NONE;

    char *temp = strdup(input);
    if (!temp) {
        fprintf(stderr, "Memory allocation failed for boolean formula in %s\n", filename);
        return -1;
    }

    char *p = trim(temp);
    char *op_pos = NULL;
    if ((op_pos = strstr(p, ">="))) {
        out->op = REL_OP_GTE;
        *op_pos = '\0';
        out->lhs = strdup(trim(p));
        out->rhs = strdup(trim(op_pos + 2));
    } else if ((op_pos = strstr(p, "<="))) {
        out->op = REL_OP_LTE;
        *op_pos = '\0';
        out->lhs = strdup(trim(p));
        out->rhs = strdup(trim(op_pos + 2));
    } else if ((op_pos = strstr(p, "=="))) {
        out->op = REL_OP_EQ;
        *op_pos = '\0';
        out->lhs = strdup(trim(p));
        out->rhs = strdup(trim(op_pos + 2));
    } else if ((op_pos = strstr(p, ">"))) {
        out->op = REL_OP_GT;
        *op_pos = '\0';
        out->lhs = strdup(trim(p));
        out->rhs = strdup(trim(op_pos + 1));
    } else if ((op_pos = strstr(p, "<"))) {
        out->op = REL_OP_LT;
        *op_pos = '\0';
        out->lhs = strdup(trim(p));
        out->rhs = strdup(trim(op_pos + 1));
    } else if ((op_pos = strstr(p, "!="))) {
        out->op = REL_OP_NEQ;
        *op_pos = '\0';
        out->lhs = strdup(trim(p));
        out->rhs = strdup(trim(op_pos + 2));
    } else {
        fprintf(stderr, "Invalid operator in boolean formula: %s in %s\n", p, filename);
        free(temp);
        return -1;
    }

    if (!out->lhs || !out->rhs) {
        fprintf(stderr, "Failed to parse boolean formula: %s in %s\n", p, filename);
        free(temp);
        free(out->lhs);
        free(out->rhs);
        return -1;
    }

    free(temp);
    return 0;
}

int temporal_load(const char *filename, TemporalFormula *out) {
    memset(out, 0, sizeof(TemporalFormula));
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return -1;
    }

    char *line = NULL;
    size_t len = 0;
    char *section = NULL;
    while (getline(&line, &len, f) != -1) {
        char *s = trim(line);
        if (!s[0] || s[0] == '#') continue;

        if (s[0] == '[') {
            s[strlen(s)-1] = '\0';
            section = strdup(s + 1);
            if (!section) {
                fclose(f); temporal_free(out); free(line); return -2;
            }
            continue;
        }

        if (!section) continue;

        if (strcmp(section, "vars") == 0) {
            if (out->var_count >= MAX_TEMPORAL_VARS) {
                fprintf(stderr, "Too many variables in %s\n", filename);
                free(section); fclose(f); temporal_free(out); free(line); return -2;
            }
            out->vars[out->var_count] = strdup(s);
            if (!out->vars[out->var_count]) {
                free(section); fclose(f); temporal_free(out); free(line); return -2;
            }
            out->var_count++;
        } else if (strcmp(section, "formula") == 0) {
            if (out->formula) free(out->formula);
            out->formula = strdup(s);
            if (!out->formula) {
                free(section); fclose(f); temporal_free(out); free(line); return -2;
            }
        } else if (strcmp(section, "testing") == 0) {
            if (out->testing_formula) {
                free(out->testing_formula->lhs);
                free(out->testing_formula->rhs);
                free(out->testing_formula);
            }
            out->testing_formula = (BooleanFormula*)calloc(1, sizeof(BooleanFormula));
            if (!out->testing_formula || parse_boolean_formula(s, out->testing_formula, filename) != 0) {
                fprintf(stderr, "Failed to parse testing formula in %s\n", filename);
                free(section); fclose(f); temporal_free(out); free(line); return -6;
            }
        } else if (strcmp(section, "memory") == 0) {
            if (out->memory_formula) free(out->memory_formula);
            out->memory_formula = strdup(s);
            if (!out->memory_formula) {
                free(section); fclose(f); temporal_free(out); free(line); return -2;
            }
        } else if (strcmp(section, "params") == 0) {
            char *key = strtok(s, " \t");
            char *value = strtok(NULL, " \t");
            if (key && value) {
                if (strcmp(key, "history_length") == 0) {
                    out->history_length = atoi(value);
                    if (out->history_length < 1 || out->history_length > 100) {
                        fprintf(stderr, "Invalid history_length in %s\n", filename);
                        free(section); fclose(f); temporal_free(out); free(line); return -3;
                    }
                } else if (strncmp(key, "k", 1) == 0 && strstr(key, "_init")) {
                    if (out->param_count >= MAX_LEARNABLE_PARAMS) {
                        fprintf(stderr, "Too many learnable parameters in %s\n", filename);
                        free(section); fclose(f); temporal_free(out); free(line); return -4;
                    }
                    out->param_names[out->param_count] = strdup(key);
                    if (!out->param_names[out->param_count]) {
                        free(section); fclose(f); temporal_free(out); free(line); return -4;
                    }
                    out->param_inits[out->param_count] = atof(value);
                    out->param_count++;
                }
            }
        }
    }

    free(line);
    free(section);
    fclose(f);

    if (!out->formula || out->var_count == 0) {
        fprintf(stderr, "Missing formula or variables in %s\n", filename);
        temporal_free(out);
        return -5;
    }

    // Validate variables in formula and testing formula
    if (validate_variables(out->formula, (const char**)out->vars, out->var_count, filename, "formula") != 0) {
        temporal_free(out);
        return -7;
    }
    if (out->testing_formula) {
        if (validate_variables(out->testing_formula->lhs, (const char**)out->vars, out->var_count, filename, "testing formula LHS") != 0 ||
            validate_variables(out->testing_formula->rhs, (const char**)out->vars, out->var_count, filename, "testing formula RHS") != 0) {
            temporal_free(out);
            return -7;
        }
    }
    if (out->memory_formula) {
        if (validate_variables(out->memory_formula, (const char**)out->vars, out->var_count, filename, "memory formula") != 0) {
            temporal_free(out);
            return -7;
        }
    }

    // Backward compatibility: if no learnable parameters specified, assume single 'k'
    if (out->param_count == 0) {
        out->param_names[0] = strdup("k");
        if (!out->param_names[0]) {
            temporal_free(out);
            return -4;
        }
        out->param_inits[0] = 0.1f; // Default initial value for 'k'
        out->param_count = 1;
    }

    return 0;
}

void temporal_free(TemporalFormula *f) {
    if (!f) return;
    free(f->formula);
    free(f->memory_formula);
    if (f->testing_formula) {
        free(f->testing_formula->lhs);
        free(f->testing_formula->rhs);
        free(f->testing_formula);
        f->testing_formula = NULL;
    }
    for (int i = 0; i < f->var_count; ++i) {
        free(f->vars[i]);
        f->vars[i] = NULL;
    }
    for (int i = 0; i < f->param_count; ++i) {
        free(f->param_names[i]);
        f->param_names[i] = NULL;
    }
    memset(f, 0, sizeof(TemporalFormula));
}
