#include <stdio.h>
#include <string.h>
#include "custom_operator.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <config.sybl>\n", argv[0]);
        return 1;
    }

    Config config = {0};
    if (parse_sybl(argv[1], &config) != 0) {
        printf("Error parsing .sybl file\n");
        return 1;
    }

    printf("Training with method: %s\n", config.method);
    if (strcmp(config.method, "algebraic") == 0) {
        train_algebraic(&config.alg_config, config.epochs, config.lr);
    } else if (strcmp(config.method, "ste") == 0) {
        train_ste(&config.ste_config, config.epochs, config.lr);
    } else {
        printf("Unknown method: %s\n", config.method);
        free_config(&config);
        return 1;
    }

    free_config(&config);
    return 0;
}
