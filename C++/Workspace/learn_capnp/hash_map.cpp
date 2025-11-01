#include <algorithm>
#include <cxxabi.h>
#include <iostream>
#include <typeinfo>
#include <unordered_map>

template <typename T> void print_type()
{
    int status;
    std::cout << abi::__cxa_demangle(typeid(T).name(), 0, 0, &status) << "\n";
}
int main()
{
    std::unordered_map<std::string, std::string> hash_map = {
        {"name", "Hung"},
        {"age", "22"},
        {"job", "SE"},
    };

    // Using lambda function to print key-value pair in hash_map
    auto print_key_value = [](const auto &key, const auto &value) {
        std::cout << "Key: " << key << ", Value: " << value << std::endl;
    };

    for (const std::pair<std::string, std::string> pair : hash_map)
    {
        print_key_value(pair.first, pair.second);
    }

    // print corresponding value of specific key
    // std::cout << hash_map.at("teo") << std::endl;

    hash_map["new key"] = "new value";
    std::cout << hash_map["new key"] << std::endl;

    for (const auto &[key, value] : hash_map)
    {
        print_key_value(key, value);
    }
    hash_map.find("new key");

    std::cout << hash_map.count("new key") << std::endl;

    print_type<decltype(hash_map)::key_type>();
    print_type<decltype(hash_map)::mapped_type>();
    print_type<decltype(hash_map)::value_type>();
    std::unordered_map<std::string, int> hash_map_new;
    print_type<decltype(hash_map_new)::key_type>();
    print_type<decltype(hash_map_new)::mapped_type>();
    print_type<decltype(hash_map_new)::value_type>();
}
