#include <cstdio>

enum class ProgrammingLang {
    C,
    CPP,
    Python,
    Julia,
    Rust,
    Lua,
};

int main() {
    // numerical data type
    float float_num = 1.2331f; // float use f or F to denote
    double double_num = 1.2345678123123123123l; 
    long double long_double_num = 1.2342342L; // long double use L to denote
    printf("float num: %g, double_num: %lf, long double num: %Lf\n",
            float_num, double_num, long_double_num);
    printf("exponential with %%e: %le\n", double_num);
    // size_t type:
    size_t size_c = sizeof(char);
    size_t size_s = sizeof(short);
    size_t size_i = sizeof(int);
    size_t size_l = sizeof(long);
    size_t size_f = sizeof(float);
    size_t size_d = sizeof(double);
    printf("char: %zd\n", size_c);
    printf("short: %zd\n", size_s);
    printf("int: %zd\n", size_i);
    printf("long: %zd\n", size_l);
    printf("float: %zd\n", size_f);
    printf("double: %zd\n", size_d);

    // array

    int my_array[] = {1, 2, 3, 4, 100};
    printf("The second number in my_array is: %d", my_array[1]);

    int maximum = 0;
    for (int i = 0; i < 5; i++ ) {
        if (my_array[i] > maximum ) maximum = my_array[i]; 
    }
    printf("Maximum value in my array: %i\n", maximum);

    std::size_t length = sizeof(my_array) / sizeof(int);
    printf("Length of my array: %zd\n", length);

    // String
    char english[] = "A book holds a house of gold.";
    printf("english: %s\n", english);

    for (int i = 0; i < 26; i++) {
        english[i] = i + 65;
    }
    printf("english: %s\n", english);

    // User-defined types
    // Enumarate
    ProgrammingLang language = ProgrammingLang::Julia;
    switch (language) {
        case ProgrammingLang::Python: {
            printf("Python\n");
        } break;
        case ProgrammingLang::Julia: {
                                         printf("Julia\n");
                                     } break;
        default: {
                     printf("Unknown lang\n");
                 }
            
    }

    return 0;
}
