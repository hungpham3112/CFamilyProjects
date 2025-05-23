#include <iostream>

class Accumulator
{
  private:
    int m_value{0};

  public:
    Accumulator() {};

    Accumulator(int value) : m_value{value} {};

    int getValue() const
    {
        return m_value;
    }

    void setValue(int value)
    {
        m_value = value;
    }

    void add(int other)
    {
        m_value += other;
    }

    friend void print(Accumulator &accumulator);
};

void print(Accumulator &accumulator)
{
    std::cout << "Current value: " << accumulator.m_value << std::endl;
}

int main()
{
    Accumulator acc{5};
    print(acc);
    acc.add(5);
    print(acc);
    acc.setValue(500);
    print(acc);
    Accumulator acc1{};
    print(acc1);
    return 0;
}
