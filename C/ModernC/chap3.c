#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>


bool is_sorted(int *arr, size_t n);


void mergesort(int *arr, size_t left, size_t right);

int main()
{
    int arr[] = {9, 2, 1, 4,1};
    size_t arr_size = sizeof(arr) / sizeof(arr[0]);
    bool result = is_sorted(arr, arr_size);
    printf("%s", result ? "sorted\n" : "not sorted\n");

    mergesort(arr, 0, arr_size);

    result = is_sorted(arr, arr_size);
    printf("%s", result ? "sorted\n" : "not sorted\n");


    const char *str = "hello";
    for (size_t i = 0; str[i] != '\0'; ++i) {
        printf("str[%ld]: %c\n", i, str[i]);
    }

    int num0[] = {2, 3, 1, 4, 5};
    int *num1 = num0;
    *(num1 + 1) = 20;
    printf("sizeof num[]: %ld\nsizeof *num: %ld\n", sizeof(num0), sizeof(num1));
    for (size_t i = 0; i < sizeof(num0) / sizeof(num0[0]); ++i) {
        printf("num0[%ld]: %i\n", i, num0[i]);
    }

    char *team[] = {"abcd", "1234", "91231"};
    printf("sizeof team: %ld\n", sizeof(team));
    char **element2 = team + 1;
    (*element2)[1] = '9';

    for (size_t i = 0; i < sizeof(team) / sizeof(char *); ++i) {
        printf("%s\n", team[i]);
        for (size_t j = 0; team[i][j] != '\0'; ++j) {
            printf("%c\n", team[i][j]);
        }
    }


    // for (size_t i = 0; i < sizeof(team[2]) / sizeof(char *); ++i) {
    //     printf("num0[%ld]: %i\n", i, team[2][i]);
    // }



    
}



void merge(int *arr, size_t left, size_t mid, size_t right) {
    size_t n1 = mid - left;
    size_t n2 = right - mid;
    
    int L[n1], R[n2];
    for (size_t i = 0; i < n1; ++i) {
        L[i] = arr[left + i];
    }
    for (size_t i = 0; i < n2; ++i) {
        R[i] = arr[mid + i];
    }

    int i = 0;
    int j = 0;
    int k = left;
    while (i < n1 && j < n2) {
        if (L[i] > R[j]) {
            arr[k] = R[j];
            ++j;
        } else {
            arr[k] = L[i];
            ++i;
        }
        ++k;
    }
    while (i < n1) {
        arr[k] = L[i];
        ++k;
        ++i;
    }

    while (j < n2) {
        arr[k] = R[j];
        ++k;
        ++j;
    }
}

void mergesort(int *arr, size_t left, size_t right){
    if (right - left < 2) {
        return;
    }
    size_t mid = (left + right) / 2;

    mergesort(arr, left, mid);
    mergesort(arr, mid, right);

    merge(arr, left, mid, right);
}

bool is_sorted(int *arr, size_t n) {
    if (n == 1) {
        return true;
    }
    for (size_t i = 0; i < n - 1; ++i) {
        if (!(arr[i] <= arr[i + 1])) {
            return false;
        }
    }
    return true;
}

