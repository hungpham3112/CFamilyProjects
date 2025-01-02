#ifndef CAT_HEADER_H
#include "../include/cat.h"
#include <iostream>
#endif

Cat::Cat() {
    _age = 5;
    _name = "tom";
    std::cout << "Cat created." << std::endl;
}

Cat::Cat(int age, std::string name) {
    _age = age;
    _name = name;
}

Cat::~Cat() 
{
    std::cout << "Cat destroyed." << std::endl;
}

int Cat::age() {
    return _age;
}

void Cat::set_age(int age) {
    _age = age;
}

std::string Cat::name() {
    return _name;
}

void Cat::set_name(std::string name) {
    _name = name;
}