#include <iostream>
#include <ostream>
#include <string>

// Procedural style
struct Date
{
    int year{2000};
    int month{1};
    int day{1};
    constexpr void print() const
    {
        ++day;
        std::cout << "Date: " << day << "/" << month << "/" << year << std::endl;
    };
};

int main()
{
    constexpr Date date{2025, 3, 17};
    date.print();
}
