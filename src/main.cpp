#include <iostream>
#include <chrono>

#include <litetensor/tensor.h>
#include <litetensor/config.h>
#include <litetensor/factor_omp.h>
#include <litetensor/als_omp.h>

int main(int argc, char** argv) {
  using namespace std;
  using namespace std::chrono;
  using namespace litetensor;
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> dsec;

  Config config(argc, argv);

  high_resolution_clock::time_point init_start = Clock::now();
  RawTensor tensor(config.tensor_file);
  double init_time = duration_cast<dsec>(Clock::now() - init_start).count();
  cout << "Initialization time: " << init_time << " seconds" << "\n\n";

  OMPALSSolver solver;

  high_resolution_clock::time_point compute_start = Clock::now();
  solver.decompose(tensor, config);
  double compute_time =
          duration_cast<dsec>(Clock::now() - compute_start).count();

  cout << "\n";
  cout << "================ Time Statistics ================" << "\n";
  cout << "Computation time: " << compute_time << " seconds" << "\n";
  cout << "Total time: " << compute_time + init_time << " seconds" << "\n";

  return 0;
}
