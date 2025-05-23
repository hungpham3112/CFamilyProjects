#include <iostream>
#include <memory>
#include <string>

class Employee
{
  private:
    std::string m_name{};

  public:
    void print()
    {
        std::cout << m_name << std::endl;
    }

    // This function is dangeous, in theory
    // you can code like this, but by logic
    // you don't want to change the private member variable
    // but if you don't have const, return by reference will
    // open the gate to modify private member variable outside the class
    // and this is counterintuitive and forbidden. Instead you should use
    // const
    /*std::string &getName()*/
    /*{*/
    /*    return m_name;*/
    /*}*/
    // Here when you put const after () but return by reference string&
    // you will get compile error , the function
    // doesn't change the member variable but it returns a gate to modify it, because
    // of that counterintuitive, therefore this is kind of const correctness.
    const std::string &getName() const
    {
        return m_name;
    }
    // Best practice, just stick to convention, if m_name is std::string
    // member function should return std::string, not using auto keyword to avoid
    // ambiguous in documentation.

    void setName(const std::string &name)
    {
        m_name = name;
    }
};

int main()
{
    Employee employee{};
    employee.setName("hello");

    /*employee.getName() = "on";*/
    employee.print();
    return 0;
}
