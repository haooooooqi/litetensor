#include <litetensor/utils.h>

#include <iostream>

namespace litetensor {

namespace utils {

bool startswith(const std::string str, const std::string s) {
  int i = 0;
  while (str[i] == ' ')
    i++;
  return str.substr(i, s.length()) == s;
}


std::vector<std::string> split(const std::string str,
                               const char separator) {
  using namespace std;
  vector<string> res;

  int len = (int) str.length();
  for (int i = 0; i < len; i++) {
    // Parse a word
    while (i < len && str[i] == ' ')
      i++;
    if (i == len)
      break;

    string word;
    while (i < len && str[i] != ' ' && str[i] != separator) {
      word.push_back(str[i]);
      i++;
    }

    res.push_back(word);
    if (i == len)
      break;

    // Meet separator
    while (i < len && str[i] == ' ')
      i++;
    if (i == len)
      break;

    if (str[i] != separator) {
      cout << "No space in key or value allowed." << endl;
      return res;
    }

  }

  return res;
}

}

}
