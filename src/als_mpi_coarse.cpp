#include <mpi.h>

#include <chrono>

#include <litetensor/tensor_mpi_coarse.h>
#include <litetensor/factor_mpi_coarse.h>
#include <litetensor/als_mpi_coarse.h>


namespace litetensor {

void CoarseMPIALSSolver::solve(CoarseTensor& tensor, Config& config) {
  CoarseFactor factor(tensor, config);
  als(tensor, factor, config);
}


void CoarseMPIALSSolver::als(CoarseTensor& tensor, CoarseFactor& factor,
                             Config& config) {
  using namespace Eigen;
  using namespace std;
  using namespace std::chrono;
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> dsec;

  uint64_t rank = config.rank;
  int max_iters = config.max_iters;
  double tolerance = config.tolerance;

  int proc_id = tensor.proc_id;

  if (proc_id == 0) {
    cout << "\n=============== Decomposing Tensor ==============\n";
    cout << "Max iterations: " << max_iters << "; " << "Rank: " << rank << "; ";
    cout << "Tolerance: " << tolerance << ";\n";
  }

  factor.ATA = factor.A.transpose() * factor.A;
  factor.BTB = factor.B.transpose() * factor.B;
  factor.CTC = factor.C.transpose() * factor.C;

  // Intermediate result of pseudo-inverse, a small dense matrix
  Mat V = MatrixXd(rank, rank);

  double prev_fitness = 1;
  double fitness = 0;

  high_resolution_clock::time_point iter_start;
  double iter_time;

  for (int iter = 0; iter < max_iters; iter++) {
    iter_start = Clock::now();
    als_iter(tensor, factor, V, iter);
    iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();

    // Master node checks fitness
    fitness = calc_fitness(factor);

    cout << "From process " << proc_id;
    cout << "Iteration time: " << iter_time << "\n";

    if (proc_id == 0)
      cout << "Iteration: " << iter + 1 << ", Fitness: " << fitness << endl;

    if (fitness == 1. || abs(fitness - prev_fitness) < tolerance)
      break;

    prev_fitness = fitness;
  }

}


void CoarseMPIALSSolver::als_iter(CoarseTensor &tensor, CoarseFactor &factor,
                                  Mat &V, int iter) {

  using namespace std;
  using namespace std::chrono;
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> dsec;

  // TODO: modify to MPI

  int proc_id = tensor.proc_id;

  // Update A
  high_resolution_clock::time_point iter_start = Clock::now();
  V = (factor.BTB.cwiseProduct(factor.CTC).llt().solve(factor.ID));
  double iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "From process " << proc_id;
  cout << "Inverse A time: " << iter_time << " seconds; ";

  iter_start = Clock::now();
  mttkrp_MA(tensor, factor, 0);
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "From process " << proc_id;
  cout << "MTTKRP A time: " << iter_time << " seconds" << std::endl;

  // Allgatherv on MA
  MPI_Allgatherv(factor.MA.data(), tensor.counts[0][proc_id], MPI_DOUBLE,
                 factor.A.data(), &tensor.counts[0].front(),
                 &tensor.disps[0].front(), MPI_DOUBLE, MPI_COMM_WORLD);

  factor.A = factor.A * V;

  normalize(factor, factor.A, iter);
  factor.ATA = factor.A.transpose() * factor.A;

  // Update B
  iter_start = Clock::now();
  V = (factor.ATA.cwiseProduct(factor.CTC).llt().solve(factor.ID));
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "From process " << proc_id;
  cout << "Inverse B time: " << iter_time << " seconds; ";

  iter_start = Clock::now();
  mttkrp_MB(tensor, factor, 1);
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "From process " << proc_id;
  cout << "MTTKRP B time: " << iter_time << " seconds" << std::endl;

  // Allgatherv on MB
  MPI_Allgatherv(factor.MB.data(), tensor.counts[1][proc_id], MPI_DOUBLE,
                 factor.B.data(), &tensor.counts[1].front(),
                 &tensor.disps[1].front(), MPI_DOUBLE, MPI_COMM_WORLD);

  factor.B = factor.B * V;

  normalize(factor, factor.B, iter);
  factor.BTB = factor.B.transpose() * factor.B;

  // Update C
  iter_start = Clock::now();
  V = (factor.ATA.cwiseProduct(factor.BTB).llt().solve(factor.ID));
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "From process " << proc_id;
  cout << "Inverse C time: " << iter_time << " seconds; ";

  iter_start = Clock::now();
  mttkrp_MC(tensor, factor, 2);
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "From process " << proc_id;
  cout << "MTTKRP C time: " << iter_time << " seconds" << std::endl;

  // Allgatherv on MC
  MPI_Allgatherv(factor.MC.data(), tensor.counts[2][proc_id], MPI_DOUBLE,
                 factor.C.data(), &tensor.counts[2].front(),
                 &tensor.disps[2].front(), MPI_DOUBLE, MPI_COMM_WORLD);

  // Copy full MC, will used in fitness calculation
  std::copy(factor.C.data(), factor.C.data() + factor.C.size(),
            factor.MC_copy.data());

  factor.C = factor.C * V;

  normalize(factor, factor.C, iter);
  factor.CTC = factor.C.transpose() * factor.C;

}


// MTTKRP for mode 1
void CoarseMPIALSSolver::mttkrp_MA(CoarseTensor& tensor, CoarseFactor& factor,
                                   uint64_t mode) {
  // Initialize MA to 0s, very important
  factor.MA.setZero();

  for (uint64_t i = 0; i < tensor.num_rows[0]; i++) {   // Each row of MA(i, :)
    for (uint64_t idx = 0; idx < tensor.indices[mode][i].size(); idx++) {
      uint64_t j = tensor.indices[mode][i][idx] % tensor.J;
      uint64_t k = tensor.indices[mode][i][idx] / tensor.J;
      factor.MA.row(i) += tensor.vals[mode][i][idx] *
              (factor.B.row(j).cwiseProduct(factor.C.row(k)));
    }
  }
}


// MTTKRP for mode 2
void CoarseMPIALSSolver::mttkrp_MB(CoarseTensor& tensor, CoarseFactor& factor,
                                   uint64_t mode) {
  factor.MB.setZero();

  for (uint64_t j = 0; j < tensor.num_rows[1]; j++) {
    for (uint64_t idx = 0; idx < tensor.indices[mode][j].size(); idx++) {
      uint64_t i = tensor.indices[mode][j][idx] % tensor.I;
      uint64_t k = tensor.indices[mode][j][idx] / tensor.I;
      factor.MB.row(j) += tensor.vals[mode][j][idx] *
              (factor.C.row(k).cwiseProduct(factor.A.row(i)));
    }
  }
}


// MTTKRP for mode 3
void CoarseMPIALSSolver::mttkrp_MC(CoarseTensor& tensor, CoarseFactor& factor,
                                   uint64_t mode) {
  factor.MC.setZero();

  for (uint64_t k = 0; k < tensor.num_rows[2]; k++) {
    for (uint64_t idx = 0; idx < tensor.indices[mode][k].size(); idx++) {
      uint64_t i = tensor.indices[mode][k][idx] % tensor.I;
      uint64_t j = tensor.indices[mode][k][idx] / tensor.I;
      factor.MC.row(k) += tensor.vals[mode][k][idx] *
              (factor.B.row(j).cwiseProduct(factor.A.row(i)));
    }
  }
}


// Mat M: A/B/C
void CoarseMPIALSSolver::normalize(CoarseFactor& factor, Mat& M, int iter) {
  uint64_t rank = factor.rank;

  if (iter == 0) {   // L2 norm in the first iteration
    for (uint64_t r = 0; r < rank; r++)
      M.col(r).normalize();
  } else {           // Max norm for later iterations
    for (uint64_t r = 0; r < rank; r++) {
      factor.lambda(r) = std::max(M.col(r).maxCoeff(), 1.0);
      M.col(r) /= factor.lambda(r);
    }
  }
}


double CoarseMPIALSSolver::calc_kruskal_norm(CoarseFactor& factor) {
  Mat tmp = factor.ATA.cwiseProduct(factor.BTB).cwiseProduct(factor.CTC);
  Mat res = factor.lambda.transpose() * tmp * factor.lambda;
  return *res.data();
}

double CoarseMPIALSSolver::calc_kruskal_inner(CoarseFactor& factor) {
  Mat tmp = factor.MC_copy.cwiseProduct(factor.C);
  Mat res = factor.ones.transpose() * tmp * factor.lambda;
  return *res.data();
}


double CoarseMPIALSSolver::calc_fitness(CoarseFactor& factor) {
  double kruskal_norm = calc_kruskal_norm(factor);
  double kruskal_inner = calc_kruskal_inner(factor);

  double residual = factor.frob_norm + kruskal_norm - (2 * kruskal_inner);

  if (residual > 0.0)
    residual = sqrt(residual);

  return 1 - (residual / factor.frob_norm_sq);
}




} // litetensor