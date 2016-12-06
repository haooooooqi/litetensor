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
  max_iters = 5;
  num_threads = 1;
  tolerance = 1e-5;

  // MPI options
  use_mpi = false;

  using namespace std;
  // Parse command line arguments
  int opt;

  while ((opt = getopt(argc, argv, "i:t:r:n")) != -1) {
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
      case 'n':
        // This also indicates that we want to use MPI
        use_mpi = true;
        break;
      case '?':
        if (optopt == 'i')
          cout << "Path to input tensor file." << endl;
        else if (optopt == 't')
          cout << "Number of threads." << endl;
        else if (optopt == 'r')
          cout << "Decompose rank." << endl;
        else if (optopt == 'n')
          cout << "Use MPI." << endl;
        break;

      default:
        cout << "Unknown options." << endl;
    }

  }


}


}
