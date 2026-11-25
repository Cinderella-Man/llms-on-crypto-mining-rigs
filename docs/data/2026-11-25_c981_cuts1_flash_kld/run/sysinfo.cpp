// Prints llama.cpp's system info line (each backend's features, Miner Llama's card-path switches among them) as the
// environment sets them, without a model: the check that the forced compute-6.1 paths are on for a run.
#include "llama.h"

#include <cstdio>

int main() {
    llama_backend_init();
    printf("%s\n", llama_print_system_info());
    llama_backend_free();
    return 0;
}
