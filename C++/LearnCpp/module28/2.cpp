#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

// mô phỏng command cat
void process_input(std::istream &stream)
{
    std::string str;
    while (std::getline(stream, str))
    {
        std::cout << str << "\n";
    }
    if (stream.eof())
    {
        std::cerr << "Stream closed by user (EOF reached)\n";
    }
}

int main(int argc, char *argv[])
{

    // {
    //
    //     char buf[10]{};
    //     std::cin >> std::setw(10) >> buf;
    //     std::cout << buf << "\n";
    //
    //     std::string line{};
    //     std::cout << line;
    // }
    {
        std::cout << argc << std::endl;
        std::cout << argv[0] << std::endl;
        if (argc > 1)
        {

            std::ifstream file{argv[1]};
            std::cout << argv[1] << std::endl;
            if (!file)
                std::cerr << "Missing file\n";
            else
                process_input(file);
        }
        else
        {
            process_input(std::cin);
        }
    }
    return 0;
}
