#ifndef UTILS_H
#define UTILS_H

#include <string>

namespace utils
{

std::string demangle(const std::string &mangled_name);
/* std::string demangle(const char *mangled_name); */
} // namespace utils

#endif
