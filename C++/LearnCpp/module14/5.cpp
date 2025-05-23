#include <iostream>

class Date
{
  public:
    Date()
    {
        std::cout << "Default constructor is called" << std::endl;
    };

    Date(int day, int month, int year) : m_day{day}, m_month(month), m_year(year)
    {
        std::cout << "Parameters constructor is called" << std::endl;
    };

    ~Date()
    {
        std::cout << "Deconstructor is called" << std::endl;
    };

    void print() const
    {
        std::cout << m_day << '/' << m_month << '/' << m_year << std::endl;
    }

  private:
    int m_day{1};
    int m_month{1};
    int m_year{1111};
};

class Point3D
{
  private:
    int m_x{};
    int m_y{};
    int m_z{};

  public:
    void setValues(int x, int y, int z)
    {
        m_x = x;
        m_y = y;
        m_z = z;
    }

    void print()
    {
        std::cout << "Point: <" << m_x << ", " << m_y << ", " << m_z << ">" << std::endl;
    }

    bool isEqual(Point3D &other)
    {
        return (m_x == other.m_x && m_y == other.m_y && m_z == other.m_z);
    }
};

int main()
{
    /*{*/
    /*    Date date;*/
    /*    date.print();*/
    /*    return 0;*/
    /*}*/

    {
        Point3D point1{};
        point1.setValues(1, 2, 3);

        Point3D point2{};
        point2.setValues(1, 2, 3);

        Point3D point3{};
        point3.setValues(4, 2, 3);

        std::cout << (point1.isEqual(point2) ? "equal\n" : "not equal\n");
        std::cout << (point1.isEqual(point3) ? "equal\n" : "not equal\n");
    }
}
