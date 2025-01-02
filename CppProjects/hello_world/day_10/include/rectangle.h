#ifndef RECTANGLE_HEADER_H
#define RECTANGLE_HEADER_H


class Rectangle {
    public:
        Rectangle(int width = 4, int height = 5);
        ~Rectangle();
        
        int get_width();
        int get_height();
        
        void set_width(int width);
        void set_height(int height);
        
        // overload class function draw_shape
        void draw_shape();
        void draw_shape(int width, int height);
    private:
        int _width;
        int _height;
};

#endif