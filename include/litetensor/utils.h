#ifndef LITETENSOR_UTILS_H
#define LITETENSOR_UTILS_H

#include <string>
#include <vector>

namespace litetensor {

namespace utils {

// String operations
bool startswith(const std::string str, const std::string s);
std::vector<std::string> split(const std::string str,
                               const char separator);

}

}


#endif //LITETENSOR_UTILS_H
