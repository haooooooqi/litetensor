#include <math.h>
#include <omp.h>

#include <chrono>
#include <iomanip>

#include <litetensor/tensor.h>
#include <litetensor/als_omp.h>

namespace litetensor {

// MTTKRP for mode 1
void OMPALSSolver::mttkrp_MA(RawTensor& tensor, Factor& factor, uint64_t mode) {
  // Initialize MA to 0s, very important
  factor.MA.setZero();

  #pragma omp parallel for schedule(dynamic, 16) num_threads(factor.num_threads)
  for (uint64_t i = 0; i < tensor.I; i++) {   // Each row of MA(i, :)
    for (uint64_t idx = 0; idx < tensor.indices[mode][i].size(); idx++) {
      uint64_t j = tensor.indices[mode][i][idx] % tensor.J;
      uint64_t k = tensor.indices[mode][i][idx] / tensor.J;
      factor.MA.row(i) += tensor.vals[mode][i][idx] *
              (factor.B.row(j).cwiseProduct(factor.C.row(k)));
    }
  }
}

// MTTKRP for mode 2
void OMPALSSolver::mttkrp_MB(RawTensor& tensor, Factor& factor, uint64_t mode) {
  factor.MB.setZero();

  #pragma omp parallel for schedule(dynamic, 16) num_threads(factor.num_threads)
  for (uint64_t j = 0; j < tensor.J; j++) {
    for (uint64_t idx = 0; idx < tensor.indices[mode][j].size(); idx++) {
      uint64_t i = tensor.indices[mode][j][idx] % tensor.I;
      uint64_t k = tensor.indices[mode][j][idx] / tensor.I;
      factor.MB.row(j) += tensor.vals[mode][j][idx] *
              (factor.C.row(k).cwiseProduct(factor.A.row(i)));
    }
  }
}

// MTTKRP for mode 3
void OMPALSSolver::mttkrp_MC(RawTensor& tensor, Factor& factor, uint64_t mode) {
  factor.MC.setZero();

  #pragma omp parallel for schedule(dynamic, 2) num_threads(factor.num_threads)
  for (uint64_t k = 0; k < tensor.K; k++) {
    for (uint64_t idx = 0; idx < tensor.indices[mode][k].size(); idx++) {
      uint64_t i = tensor.indices[mode][k][idx] % tensor.I;
      uint64_t j = tensor.indices[mode][k][idx] / tensor.I;
      factor.MC.row(k) += tensor.vals[mode][k][idx] *
              (factor.B.row(j).cwiseProduct(factor.A.row(i)));
    }
  }
}


void OMPALSSolver::normalize(Factor& factor, Mat& M, int iter) {
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


void OMPALSSolver::als_iter(RawTensor& tensor, Factor& factor, Mat& V,
                            int iter) {
  using namespace std;
  using namespace std::chrono;
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> dsec;

  high_resolution_clock::time_point iter_start;
  double iter_time;
  int width = 8;

  // Update A
  V = (factor.BTB.cwiseProduct(factor.CTC).llt().solve(factor.ID));

  iter_start = Clock::now();
  mttkrp_MA(tensor, factor, 0);
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "A MTTKRP time: " << setw(width) << iter_time << " seconds; ";

  iter_start = Clock::now();
  factor.A = factor.MA * V;
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "Update time: " << setw(width) << iter_time << " seconds; ";

  iter_start = Clock::now();
  normalize(factor, factor.A, iter);
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "Normalize time: " << setw(width) << iter_time << " seconds; ";

  iter_start = Clock::now();
  factor.ATA = factor.A.transpose() * factor.A;
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "ATA time: " << setw(width) << iter_time << " seconds;\n";

  // Update B
  V = (factor.ATA.cwiseProduct(factor.CTC).llt().solve(factor.ID));

  iter_start = Clock::now();
  mttkrp_MB(tensor, factor, 1);
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  std::cout << "B MTTKRP time: " << setw(width) << iter_time << " seconds; ";

  iter_start = Clock::now();
  factor.B = factor.MB * V;
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "Update time: " << setw(width) << iter_time << " seconds; ";

  iter_start = Clock::now();
  normalize(factor, factor.B, iter);
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "Normalize time: " << setw(width) << iter_time << " seconds; ";

  iter_start = Clock::now();
  factor.BTB = factor.B.transpose() * factor.B;
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "BTB time: " << setw(width) << iter_time << " seconds;\n";

  // Update C
  V = (factor.ATA.cwiseProduct(factor.BTB).llt().solve(factor.ID));

  iter_start = Clock::now();
  mttkrp_MC(tensor, factor, 2);
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "C MTTKRP time: " << setw(width) << iter_time << " seconds; ";

  iter_start = Clock::now();
  factor.C = factor.MC * V;
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "Update time: " << setw(width) << iter_time << " seconds; ";

  iter_start = Clock::now();
  normalize(factor, factor.C, iter);
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "Normalize time: " << setw(width) << iter_time << " seconds; ";

  iter_start = Clock::now();
  factor.CTC = factor.C.transpose() * factor.C;
  iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();
  cout << "CTC time: " << setw(width) << iter_time << " seconds;\n";
}


void OMPALSSolver::als(RawTensor& tensor, Factor& factor, Config& config) {
  using namespace Eigen;
  using namespace std;
  using namespace std::chrono;
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> dsec;

  uint64_t rank = config.rank;
  int num_threads = config.num_threads;
  int max_iters = config.max_iters;
  double tolerance = config.tolerance;

  cout << "====================== Decomposing Tensor ======================\n";
  cout << "Max iterations: " << max_iters << "; " <<  "Rank: " << rank << "; ";
  cout << "Tolerance: " << tolerance << "; ";
  cout << "Number of threads: " << num_threads << endl;

  factor.ATA = factor.A.transpose() * factor.A;
  factor.BTB = factor.B.transpose() * factor.B;
  factor.CTC = factor.C.transpose() * factor.C;

  /*
  cout << "Initial A, B, C\n";
  cout << "A: \n" << factor.A << endl;
  cout << "B: \n" << factor.B << endl;
  cout << "C: \n" << factor.C << endl;
   */

  Mat V = MatrixXd(rank, rank);

  double prev_fitness = 1;
  double fitness = 0;
  high_resolution_clock::time_point iter_start;
  double iter_time;

  for (int iter = 0; iter < max_iters; iter++) {
    iter_start = Clock::now();
    als_iter(tensor, factor, V, iter);
    iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();

    // Check fitness
    fitness = calc_fitness(factor);
    cout << "Time: " << iter_time << ", ";
    cout << "Iteration: " << iter + 1 << ", Fitness: " << fitness << endl;

    if (fitness == 1. || abs(fitness - prev_fitness) < tolerance)
      break;

    prev_fitness = fitness;
  }

}


void OMPALSSolver::decompose(RawTensor& tensor, Config& config) {
  Factor factor(tensor, config);
  als(tensor, factor, config);
}

/*
 * Fitness functions
 */
double OMPALSSolver::calc_kruskal_norm(Factor& factor) {
  Mat tmp = factor.ATA.cwiseProduct(factor.BTB).cwiseProduct(factor.CTC);
  Mat res = factor.lambda.transpose() * tmp * factor.lambda;
  return *res.data();
}

double OMPALSSolver::calc_kruskal_inner(Factor& factor) {
  Mat tmp = factor.MC.cwiseProduct(factor.C);
  Mat res = factor.ones.transpose() * tmp * factor.lambda;
  return *res.data();
}


double OMPALSSolver::calc_fitness(Factor& factor) {
  double kruskal_norm = calc_kruskal_norm(factor);
  double kruskal_inner = calc_kruskal_inner(factor);

  double residual = factor.frob_norm + kruskal_norm - (2 * kruskal_inner);

  if (residual > 0.0)
    residual = sqrt(residual);

  return 1 - (residual / factor.frob_norm_sq);
}

}  // namespace litetensor
