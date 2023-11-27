#include <iostream>
#include <cstdio>
#include <typeinfo>


struct ProgrammingLang {
    char name[256];
};

void print_names(ProgrammingLang* names, size_t length) {
    for (size_t i = 0; i < length; i++) {
        printf("%s is awesome\n", names[i].name);
    }

}

int main(int argc, char *argv[])
{
    int gettysburg{};
    printf("gettysburg: %d\n", gettysburg);
    int* gettysburg_address = &gettysburg;
    printf("&gettysburg_address is: %p\n", gettysburg_address);
    *gettysburg_address = 5;
    printf("The value of gettysburg_address: %d\n", gettysburg);
    printf("The value of gettysburg_address: %p\n", &gettysburg);

    int arr[]{1, 2, 3};
    int* arr_address = arr;
    printf("arr: %p, arr_address: %p\n", arr, arr_address);
    ProgrammingLang names[]{{"Python"}, {"C"}, {"C++"}, {"Julia"}};
    print_names(names, sizeof(names)/ sizeof(ProgrammingLang));

    // int* ptr = &gettysburg;
    // std::cout << ptr << std::endl;
    int& ptr = gettysburg;
    int* ptr_of_ptr = &ptr;
    std::cout << ptr << std::endl;
    ptr = 7;
    std::cout << "ptr: " << ptr << "gettysburg: " << gettysburg<< std::endl;
    printf("%p\n", ptr_of_ptr);
    printf("%p\n", &gettysburg);

    std::cout << "Type of gettysburg: " << typeid(gettysburg).name() << std::endl;
    return 0;
}
