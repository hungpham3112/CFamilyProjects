#include <iostream>

// int *dangling_ref()
// {
//     int x = 42;
//     return &x; // returns pointer to local stack variable
// }

int main()
{
    // {
    //     int *p = dangling_ref();
    //
    //     std::cout << *p << "\n"; // UB: dangling pointer
    // }

    {

        int *p = new int(42);
        delete p;
        std::cout << *p << "\n"; // UB: use-after-free
    }
    // {
    //     std::string name{"hung"};
    //     std::string another_name = name;
    //     std::cout << name << std::endl;
    // }

    // {
    //     int *p = new int(5);
    //     delete p;
    //     delete p; // UB: double free
    // }
}
