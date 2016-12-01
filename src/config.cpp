#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>

#include <string>
#include <iostream>

#include <litetensor/config.h>

namespace litetensor {

Config::Config(int argc, char** argv) {
  // Default value
  rank = 10;
  max_iters = 10;
  num_threads = 1;
  tolerance = 1e-5;

  using namespace std;
  // Parse command line arguments
  int opt;

  while ((opt = getopt(argc, argv, "i:t:r:")) != -1) {
    switch (opt) {
      case 'i':
        tensor_file = string(optarg);
        break;
      case 't':
        num_threads = atoi(optarg);
        break;
      case 'r':
        rank = (uint64_t) atoi(optarg);
        break;
      case '?':
        if (optopt == 'i')
          cout << "Path to input tensor file." << endl;
        else if (optopt == 't')
          cout << "Number of threads." << endl;
        else if (optopt == 'r')
          cout << "Decompose rank." << endl;
        break;
    }

  }


}


}
