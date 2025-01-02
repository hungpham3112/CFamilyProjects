#include <iostream>
#include <ostream>
#include "../include/cat.h"

int main() {
    // {
    //     int arr[2][4] {0, 1, 2, 3, 4, 5, 6 ,7 };
    //     for (int i = 0; i < 2; ++i) {
    //         for (int j = 1; j < 4; ++j) {
    //             std::cout << arr[i][j] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }
    {
        // Cat group_of_cat[4];
        // for (int i = 0; i < 4; ++i) {
        //     group_of_cat[i].set_age(i * 2);
        // }

        // for (int i = 0; i < 4; ++i) {
        //     std::cout << "cat number: " << i 
        //             << " Age: " << group_of_cat[i].age() << std::endl;
        // }

        // std::cout << "Use raw object on stack cost: " << sizeof(group_of_cat) << " bytes\n";
        
        // Cat *second_group_of_cat[4];
        // Cat *cat_ptr;
        // for (int i = 0; i < 4; ++i) {
        //     cat_ptr = new Cat;
        //     cat_ptr->set_age(i + 2);
        //     second_group_of_cat[i] = cat_ptr;
        // }

        // for (int i = 0; i < 4; ++i) {
        //     std::cout << "cat number: " << i 
        //             << " Age: " << group_of_cat[i].age() << std::endl;
        // }

        // std::cout << "Use pointer on stack cost: " << sizeof(second_group_of_cat) << " bytes";
        
        // Cat * group_of_cat_ptr = new Cat[4];

        // // std::cout << "Use pointer on stack cost: " << sizeof(second_group_of_cat) << " bytes";
        
        // delete [] group_of_cat_ptr;
        // delete [] cat_ptr;
    }
    // {
    //     char buffer[6];
    //     std::cout << "Enter the string: " << std::endl;
    //     std::cin.get(buffer, 5);
    //     std::cout << buffer << std::endl;
    // }
    return 0;
}