#include <iostream>
#include <chrono>

#include <litetensor/tensor.h>
#include <litetensor/config.h>
#include <litetensor/als_sequential.h>
#include <litetensor/als_omp.h>


int main(int argc, char** argv) {
  using namespace std;
  using namespace std::chrono;
  using namespace litetensor;
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> dsec;

  Config config(argc, argv);

  high_resolution_clock::time_point init_start = Clock::now();
  SequentialTensor tensor(config.tensor_file);
  double init_time = duration_cast<dsec>(Clock::now() - init_start).count();

  OMPALSSolver solver(tensor, config.rank, config.num_threads);

  high_resolution_clock::time_point compute_start = Clock::now();
  solver.decompose(tensor, config.max_iters, config.tolerance);
  double compute_time =
          duration_cast<dsec>(Clock::now() - compute_start).count();

  cout << endl;
  cout << "================ Time Statistics ================" << endl;
  cout << "Initialization time: " << init_time << " seconds" << endl;
  cout << "Computation time: " << compute_time << " seconds" << endl;
  cout << "Total time: " << compute_time + init_time << " seconds" << endl;

//  solver.check_correctness(tensor);

  return 0;
}
