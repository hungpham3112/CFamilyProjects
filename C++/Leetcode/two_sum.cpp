#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

std::vector<int> twoSum(std::vector<int> &vec, int target)
{

    std::unordered_map<int, int> hash_map;
    for (int i = 0; i < vec.size(); ++i)
    {
        if (hash_map.find(target - vec[i]) != hash_map.end())
        {
            return {hash_map[target - vec[i]], i};
        }
        hash_map[vec[i]] = i;
    }

    return {};
}

int main()
{
    std::string line;
    std::vector<int> vec;
    int target;

    std::cout << "Enter vector: ";
    std::getline(std::cin, line);
    std::stringstream ss(line);
    int x;
    while (ss >> x)
        vec.push_back(x);

    std::cout << "Enter target: ";
    std::cin >> target;

    std::cout << "Vector: ";
    for (int i : vec)
        std::cout << i << " ";
    std::cout << "\nTarget: " << target << "\n";

    for (auto &result : twoSum(vec, target))
    {
        std::cout << result << std::endl;
    }
}
