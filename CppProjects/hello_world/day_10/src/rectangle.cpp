#include "../include/rectangle.h"
#include <iostream>

// Here we use initialization list
Rectangle::Rectangle(int width, int height) : _width(width), _height(height)
{
}

// Here we use assignment in the body
// Rectangle::Rectangle(int width, int height)
// {
//     _width = width;
//     _height = height;
// }

Rectangle::~Rectangle(){};

void Rectangle::set_height(int height)
{
    _height = height;
}

void Rectangle::set_width(int width)
{
    _width = width;
}

int Rectangle::get_height()
{
    return _height;
}

int Rectangle::get_width()
{
    return _width;
}

void Rectangle::draw_shape()
{
    draw_shape(_width, _height);
}

void Rectangle::draw_shape(int width, int height)
{
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            std::cout << "*";
        }
        std::cout << std::endl;
    }
}
