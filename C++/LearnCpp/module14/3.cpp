#include <iostream>
#include <ostream>

/*struct Person*/
/*{*/
/*    std::string name{default_name};*/
/*    int age{};*/
/*    void kisses(const Person *person)*/
/*    {*/
/*        std::cout << this->name << " kisses " << person->name << std::endl;*/
/*    }*/
/*    std::string default_name{"hung"};*/
/**/
/*    void kisses(const std::string &moment)*/
/*    {*/
/*        std::cout << moment << " " << name << " kiss " << std::endl;*/
/*    }*/
/*};*/

// Question 1:

// Provide the definition for IntPair and the print() member function here

struct IntPair
{
    int x{};
    int y{};
    void print()
    {
        std::cout << "Pair(" << x << ", " << y << ")" << std::endl;
    }
    bool isEqual(const IntPair &other)
    {
        return (x == other.x) && (y == other.y);
    }
};

int main()
{
    IntPair p1{1, 2};
    IntPair p2{3, 4};

    std::cout << "p1: ";
    p1.print();

    std::cout << "p2: ";
    p2.print();

    std::cout << "p1 and p1 " << (p1.isEqual(p1) ? "are equal\n" : "are not equal\n");
    std::cout << "p1 and p2 " << (p1.isEqual(p2) ? "are equal\n" : "are not equal\n");
}
