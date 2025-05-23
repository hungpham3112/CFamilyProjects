#include <iostream>
#include <optional>

class Point2D
{
  private:
    int m_x{};
    int m_y{};

  public:
    Point2D(int x, int y) : m_x{x}, m_y{y}
    {
        std::cout << "Point2D is constructed: Point<" << m_x << ", " << m_y << ">" << std::endl;
    }

    ~Point2D() {};

    void print()
    {
        std::cout << "Point<" << m_x << ", " << m_y << ">" << std::endl;
    }
};

class Fraction
{
  private:
    int m_numerator;
    int m_denominator;

  public:
    Fraction() {};
    Fraction(int numerator, int denominator) : m_numerator{numerator}, m_denominator{denominator} {};

    void print() {
        std::cout << "Fraction " << m_numerator << ", "<< m_denominator << std::endl;
    }

    friend std::optional<Fraction> createFraction(int numerator, int denominator);
};

std::optional<Fraction> createFraction(int numerator, int denominator)
{
    if (denominator == 0)
    {
        return std::nullopt;
    }
    else
    {
        return Fraction(numerator, denominator);
    }
};

int main()
{
    {
        Point2D point(4, 3);
        point.print();
    }

    {

        /* auto fraction{createFraction(1, 1)}; */
        /* if (fraction) { */
/* fraction.print() */
        /* } else { */
        /* std::cout << "ngu" << std::endl; */
        /* } */
   auto f1 { createFraction(0, 1) };
    if (f1)
    {
        std::cout << "Fraction created\n";
    }

    auto f2 { createFraction(0, 0) };
    if (!f2)
    {
        std::cout << "Bad fraction\n";
    }
    }
    return 0;
}
