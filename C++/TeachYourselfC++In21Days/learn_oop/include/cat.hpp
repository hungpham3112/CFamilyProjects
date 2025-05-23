#pragma once
#include <string>


enum class Gender {
    Male,
    Female
};

typedef unsigned short int USINT;

class Cat {
 public:
  Cat(USINT init_age, Gender init_gender);
  ~Cat();

  Gender gender();
  void set_gender(Gender cat_gender);

  int age();
  // why you set const before Gender to avoid  conflict?
  // I don't understand the error, please take a look in future.
  // https://stackoverflow.com/questions/18565167/non-const-lvalue-references-cannot-be-bound-to-an-lvalue-of-different-type
  void set_age(USINT cat_age);

  std::string& name();
  void set_name(std::string cat_name);

  void meow();

 private:
  Gender gender_;
  std::string name_;
  USINT age_;
};