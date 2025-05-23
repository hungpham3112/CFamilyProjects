#include <iostream>
#include <string>

class Member
{
  public:
    Member(const std::string &name) {};
    ~Member() {};

    std::string getName() const
    {
        return m_name;
    }

  private:
    std::string m_name;
};

void print(Member &member)
{
    std::cout << "Member name is: " << member.getName() << std::endl;
}

int main()
{
    Member member{"Hung"};
    std::cout << member.getName();
    return 0;
}
