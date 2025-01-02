#include "../include/cat.h"
#include "../include/utils.h"

#include <iostream>


int main() {
    // {
    //     using namespace std;
    //     int intOne{5};
    //     int &rSomeRef = intOne;

    //     cout << "intOne: " << intOne << endl;
    //     cout << "rSomeRef: " << rSomeRef << endl;

    //     int intTwo{7};
    //     // Here you passing value of intTwo to rSomeRef,
    //     // but rSomeRef points to intOne, so intOne = intTwo (by value) 
    //     // while the location of intOne and rSomeRef remains the same. 
    //     rSomeRef = intTwo; 

    //     cout << "intOne: " << intOne << endl;
    //     cout << "intTwo: " << intTwo << endl;
    //     cout << "rSomeRef: " << rSomeRef << endl;
    //     cout << "&intOne: " << &intOne << endl;
    //     cout << "&intTwo: " << &intTwo << endl;
    //     cout << "&rSomeRef: " << &rSomeRef << endl;
    // }

    // { 
    //     Cat tom(12, "tom");
    //     Cat &tom_ref = tom;
        
    //     // object reference can access to the target attributes identical to the target.
    //     std::cout << "tom is: " <<  tom.age() << " years old" << std::endl;
    //     std::cout << "tom_ref is: " <<  tom_ref.age() << " years old" << std::endl;
    // }

    return 0;
}