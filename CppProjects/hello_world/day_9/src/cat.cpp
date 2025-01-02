#ifndef CAT_HEADER_H
#include "../include/cat.h"
#endif

Cat::Cat(int age, std::string name) {
    age_ = age;
    name_ = name;
}

Cat::~Cat() {}

int Cat::age() {
    return age_;
}

void Cat::set_age(int age) {
    age_ = age;
}

std::string Cat::name() {
    return name_;
}

void Cat::set_name(std::string name) {
    name_ = name;
}