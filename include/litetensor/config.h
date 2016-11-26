#ifndef LITETENSOR_CONFIG_H
#define LITETENSOR_CONFIG_H

#include <litetensor/utils.h>

namespace litetensor {

/*
 * Configuration struct
 */
struct Config {
public:
  std::string tensor_file;

  uint64_t rank;
  int max_iters;
  int num_threads;
  double tolerance;

  Config(int argc, char** argv);

};


}


#endif //LITETENSOR_CONFIG_H
