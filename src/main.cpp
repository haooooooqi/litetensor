#include <mpi.h>

#include <iostream>
#include <chrono>

#include <litetensor/config.h>
#include <litetensor/tensor.h>
#include <litetensor/tensor_mpi_coarse.h>
#include <litetensor/factor_omp.h>
#include <litetensor/als_omp.h>

int main(int argc, char** argv) {
  using namespace std;
  using namespace std::chrono;
  using namespace litetensor;
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> dsec;

  Config config(argc, argv);

  if (config.use_mpi) {     // MPI code
    int num_procs;          // Number of processes
    int proc_id;            // Process ID
    int master_id = 0;      // Master node

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Master node will read tensor, partition tensor to chunk of rows
    // Send rows to each node
    if (proc_id == master_id) {
      config.num_procs = num_procs;

      Partitioner partitioner(config);

    }

    // Do ALS until convergence

    MPI_Finalize();
  } else {              // Single node, shared address model
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
  }

  return 0;
}
