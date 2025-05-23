#include <iostream>
#include <ostream>
#include <string>

// Procedural style
/*struct Date*/
/*{*/
/*    int year{2000};*/
/*    int month{1};*/
/*    int day{1};*/
/*};*/
/**/
/*// This is non-member function*/
/*void print(Date &date)*/
/*{*/
/*    std::cout << "Date: " << date.day << "/" << date.month << "/" << date.year << std::endl;*/
/*}*/

// OOP style

/*struct Date*/
/*{*/
/*    int year{2000};*/
/*    int month{1};*/
/*    int day{1};*/
/*    void print(Date &date);*/
/*};*/
/*// struct, class. union are all aggregate style*/
/*// they allow function declared inside or outside it*/
/*// This is non - member function*/
/*void Date::print(Date &date)*/
/*{*/
/*    std::cout << "Date: " << date.day << "/" << date.month << "/" << date.year << std::endl;*/
/*}*/

int main()
{
    /*Date date{2025, 3, 17};*/
    /*{*/
    /*    print(date);*/
    /*}*/
    /*{*/
    /*    date.print(date);*/
    /*}*/

    /*{*/
    /*    Person hung{"???"};*/
    /*    Person hello{"hello"};*/
    /*    std::string moment{"yesterday"};*/
    /**/
    /*    hung.kisses(&hello);*/
    /*    std::cout << hung.name;*/
    /*    hung.kisses(moment);*/
    /*}*/

    return 0;
}
