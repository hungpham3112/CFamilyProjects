// Read https://en.cppreference.com/w/cpp/language/raii
#include <iostream>
#include <typeinfo>
#include <numeric>


void pp(int a)
{
    std::cout << &a << " <- " << a << " with size: " << sizeof(a) << std::endl;
}

template <typename T> class DynamicArray
{
  public:
    DynamicArray(size_t size) : m_arr(new T[size]), m_size(size)

    {
        std::cout << "Array Constructor" << std::endl;
    }
    ~DynamicArray()
    {
        delete[] m_arr;
        std::cout << "Array Deconstructor" << std::endl;
    }
    T get(size_t idx) const
    {
        return m_arr[idx];
    }

    void set(size_t idx, T new_val) const
    {
        m_arr[idx] = new_val;
    }

    void print() const
    {
        for (size_t i = 0; i < m_size; ++i)
        {
            std::cout << i << " " << m_arr[i] << std::endl;
        }
    }

    T &operator[](size_t idx) const
    {
        return m_arr[idx];
    }

  private:
    T *m_arr;
    size_t m_size;
};

int main(int argc, char *argv[])
{
    {
        const DynamicArray<float> arr(10);
        arr.set(8, 1.9);
        std::cout << "arr[idx]: " << arr[8] << std::endl;
        arr.print();
        std::cout << "in" << std::endl;
        std::cout << typeid(arr.get(0)).name() << std::endl;
    }
    std::cout << "out" << std::endl;
    {
        int *arr = new int[10];
        std::cout << "arr[idx]: " << arr[8] << std::endl;
    }
    int arr[10];
    std::iota(std::begin(arr), std::end(arr), 10);
    std::cout << "arr[idx]: " << arr[8] << std::endl;

    int num = 9;
    pp(num);
    std::cout << &num << std::endl;


    return 0;
}
