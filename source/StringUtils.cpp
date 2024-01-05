#include "quark/StringUtils.h"

#include <sstream>

std::string trim(const std::string &input) {
  size_t first = input.find_first_not_of(' ');
  if (std::string::npos == first) {
    return input;
  }
  size_t last = input.find_last_not_of(' ');
  return input.substr(first, (last - first + 1));
}

std::vector<std::string> split(const std::string &input, const char delimiter) {
  std::vector<std::string> ret;
  std::string token;
  std::istringstream iss(input);
  while (std::getline(iss, token, delimiter)) {
    ret.push_back(token);
  }

  return ret;
}
