#include <stdio.h>

void printRawBytes(const char* str, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        printf("%02X ", (unsigned char)str[i]);
    }
    printf("\n");
}

int main() {
    char myString[] = "Hello\0World";

    // Print the raw bytes of the string
    printf("Raw bytes: ");
    printRawBytes(myString, sizeof(myString));

    return 0;
}

