#ifndef CAT_HEADER_H
#define CAT_HEADER_H
#include <string>

class Cat {
    public:
        Cat();
        Cat(int age, std::string name);
        ~Cat();
        int age();
        void set_age(int age);
        std::string name();
        void set_name(std::string name);
    
    private:
        int _age;
        std::string _name;

};

#endif