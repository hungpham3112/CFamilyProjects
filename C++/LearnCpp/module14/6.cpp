#include <iostream>

class Date
{
  private:
    int m_year{1999};
    int m_month{1};
    int m_day{1};

  public:
    void print() const
    {
        std::cout << m_year << '/' << m_month << '/' << m_day << '\n';
    }

    // Getters and setters are under the term access functions
    int getYear() const
    {
        return m_year;
    }

    void setYear(int year)
    {
        m_year = year;
    }

    int getMonth() const
    {
        return m_month;
    }

    void setMonth(int month)
    {
        m_month = month;
    }

    int getDay() const
    {
        return m_day;
    }

    void setDay(int day)
    {
        m_day = day;
    }
};

int main()
{
    Date d{}; // create a Date object
    d.setDay(4);
    d.print();

    return 0;
}
