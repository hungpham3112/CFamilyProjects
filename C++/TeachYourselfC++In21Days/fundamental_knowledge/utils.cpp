#include "include/utils.h"
#include <cxxabi.h>

std::string utils::demangle(const std::string &mangled_name)
{
    return abi::__cxa_demangle(mangled_name.c_str(), nullptr, nullptr, 0);
}
