#include <iostream>
#include <ostream>
#include <string>
#include <sys/types.h>

class Employee
{
  private:
    std::string m_name{"???"};
    u_int64_t m_id{0};
    bool m_isManager{false};

  public:
    /*Employee(std::string name = "hung", u_int64_t id = 123, bool isManager = true)*/
    Employee(std::string name, u_int64_t id, bool isManager) : m_name{name}, m_id{id}, m_isManager{isManager}
    {
        std::cout << m_name << " " << m_id << " " << m_isManager << std::endl;
    };

    Employee() : Employee("hung", 123, true)
    {
        std::cout << "Deglating constructor called" << std::endl;
    };
};

class Weapon
{
  private:
    std::string m_name;
    int m_damage;
    double m_range;

  public:
    Weapon(std::string name = "Sword", int damage = 200, double range = 5.8)
        : m_name(name), m_damage(damage), m_range(range)
    {
        if (damage < 200)
        {

            std::cout << " danh xa " << std::endl;
        }
        else if (200 < damage && damage < 300)
        {

            std::cout << " danh gan " << std::endl;
        }

        std::cout << m_name << " " << m_damage << " " << m_range << std::endl;
    };

    std::string getName() const
    {
        return m_name;
    };
    Weapon(int damage, double range) : Weapon("Sword", damage, range) {};
    Weapon(int damage) : Weapon("Sword", damage, 9) {};
};

class Character
{
  private:
    std::string m_name;
    int m_hp;
    Weapon m_weapon;

  public:
    Character(std::string name, int hp, const Weapon &weapon) : m_name(name), m_hp(hp), m_weapon(weapon) {};
    void printStatus()
    {
        std::cout << m_name << " " << m_hp << " " << m_weapon.getName() << std::endl;
    };

    virtual void attack()
    {
    }
};

class Warrior : public Character
{
    void attack() override
    {
        std::cout << "Warrior is attacking" << std::endl;
    }
};

class Archer : public Character
{
    void attack() override
    {
        std::cout << "Archer is attacking" << std::endl;
    }
};

class Mage : public Character
{
    void attack() override
    {
        std::cout << "Mage is attacking" << std::endl;
    }
};

int main()
{
    // default argument inside constructor already provide the same functionality
    // as the default constructor, so you don't need to use default constructor
    // when you already have the default argument
    Weapon sword{};
    Weapon knife{"knife", 12, 12};
    Weapon gun{"gun", 200, 500};
    Character assasin{"bob", 100, knife};
    Character ad{"violet", 100, gun};
    assasin.printStatus();
    ad.printStatus();
    return 0;
}
