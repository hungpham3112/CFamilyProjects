#include "dnd_character.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

static int dice_roll(void) {
    srand(time(NULL));
    int num = (rand() % 6) + 1;
    return num;
}

static inline void swap(int *num1, int *num2) {
    int tmp;
    if (*num1 > *num2) {
        tmp = *num1;
        *num1 = *num2;
        *num2 = tmp;
    }
}

static void sort4(int arr[static 4]) {
    swap(arr, arr + 1);
    swap(arr + 2, arr + 3);

    swap(arr, arr + 2);
    swap(arr + 1, arr + 3);
    
    swap(arr + 1, arr + 2);
}

int ability(void) {
    int arr[4];
    for (int i = 0; i < 4; ++i) {
        arr[i] = dice_roll();
    }
    sort4(arr);
    return arr[1] + arr[2] + arr[3];
}

int modifier(int score) {
    return (int) floor((score - 10) / 2.0);
}

dnd_character_t make_dnd_character(void) {
    return (dnd_character_t) {
       .strength = ability(),
       .dexterity = ability(),
       .constitution = ability(),
       .intelligence = ability(),
       .wisdom = ability(),
       .charisma = ability(),
       .hitpoints = 10 + modifier(ability()),
    };
}