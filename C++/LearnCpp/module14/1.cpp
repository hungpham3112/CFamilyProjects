#include <iostream>
#include <string_view>
#include <variant>
#include <vector>

// The first way using procedual paradigm
/*enum AnimalType*/
/*{*/
/*    Cat,*/
/*    Dog,*/
/*    Chicken,*/
/*    Snake   */
/*};*/
/**/
/*// string_view chua biet de lam gi*/
/*constexpr std::string_view animal_name(AnimalType type)*/
/*{*/
/*    switch (type)*/
/*    {*/
/*    case Cat:*/
/*        return "cat";*/
/*    case Dog:*/
/*        return "dog";*/
/*    case Chicken:*/
/*        return "chicken";*/
/*    case Snake:*/
/*        return "snake";*/
/*    default:*/
/*        return "";*/
/*    }*/
/*}*/
/**/
/*// constexpr is a way to evaluate expressions at compile-time*/
/*// when possible, to optimize performance of binary file by*/
/*// replacing expressions with constant values.*/
/*constexpr int num_legs(AnimalType type)*/
/*{*/
/*    switch (type)*/
/*    {*/
/*    case Cat:*/
/*        return 4;*/
/*    case Dog:*/
/*        return 4;*/
/*    case Chicken:*/
/*        return 2;*/
/*    case Snake:*/
/*        return 0;*/
/*    default:*/
/*        return 0;*/
/*    }*/
/*}*/

// The second way uses OOP paradigm
struct Cat
{
    std::string_view name{"cat"};
    int num_legs{4};
};

struct Dog
{
    std::string_view name{"dog"};
    int num_legs{4};
};

struct Chicken
{
    std::string_view name{"chicken"};
    int num_legs{2};
};

struct Snake
{
    std::string_view name{"snake"};
    int num_legs{0};
};

int main()
{
    /*{*/
    /*    std::vector<AnimalType> vec{};*/
    /*    constexpr AnimalType cat{Cat};*/
    /*    vec.push_back(cat);*/
    /*    constexpr AnimalType chicken{Chicken};*/
    /*    vec.push_back(chicken);*/
    /*    constexpr AnimalType dog{Dog};*/
    /*    vec.push_back(dog);*/
    /*    constexpr AnimalType snake{snake};*/
    /*    vec.push_back(snake);*/
    /*    for (AnimalType &animal : vec)*/
    /*    {*/
    /*        std::cout << animal_name(animal) << " has: " << num_legs(animal) << " legs" << std::endl;*/
    /*    }*/
    /*}*/
    {

        std::vector<std::variant<Cat, Dog, Chicken, Snake>> vec{};
        constexpr Cat cat;
        constexpr Dog dog;
        constexpr Chicken chicken;
        constexpr Chicken snake;
        vec.push_back(cat);
        vec.push_back(dog);
        vec.push_back(chicken);
        vec.push_back(snake);
        for (auto &animals : vec)
        {
            std::visit(
                [](auto &&animal) { std::cout << animal.name << " has: " << animal.num_legs << " legs" << std::endl; },
                animals);
        }
        return 0;
    }
}

// Consume 40m
