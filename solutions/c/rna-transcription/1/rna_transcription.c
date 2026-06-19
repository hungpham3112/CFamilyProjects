#include "rna_transcription.h"
#include <string.h>
#include <stdlib.h>

char *to_rna(const char *dna) {
    size_t len = strlen(dna);
    char *rna = malloc(len + 1);
    if (!rna) return NULL;
    for (size_t i = 0; dna[i] != '\0'; ++i) {
     if (dna[i] == 'G') rna[i] = 'C';
     if (dna[i] == 'C') rna[i] = 'G';
     if (dna[i] == 'T') rna[i] = 'A';
     if (dna[i] == 'A') rna[i] = 'U';
    }    
    rna[len] = '\0';
    return rna;
}

