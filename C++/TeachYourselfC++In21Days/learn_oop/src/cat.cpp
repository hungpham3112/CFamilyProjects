#include <iostream>
#include "cat.hpp"

Cat::Cat(USINT init_age, Gender init_gender ) {
    age_ = init_age;
    gender_ = init_gender;
    std::cout << "Cat constructor\n";
}

Cat::~Cat() {
    std::cout << "Cat destructor\n";
}

Gender Cat::gender() {
    return gender_;
}

void Cat::set_gender(Gender cat_gender) {
    gender_ = cat_gender;
}

int Cat::age() {
    return age_;
}

void Cat::set_age(USINT cat_age) {
    age_ = cat_age;
}

std::string& Cat::name() {
    return name_;
}

void Cat::set_name(std::string cat_name) {
    name_ = cat_name;
}

