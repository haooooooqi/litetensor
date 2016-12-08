#include <mpi.h>

#include <iomanip>

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

  uint64_t rank = config.rank;
  int max_iters = config.max_iters;
  double tolerance = config.tolerance;

  int proc_id = tensor.proc_id;

  if (proc_id == 0) {
    cout << "\n";
    cout << "====================== Decomposing Tensor =====================\n";
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

  double iter_start;
  double iter_time;
  double max_time;
  int width = 8;

  for (int iter = 0; iter < max_iters; iter++) {
    iter_start = MPI_Wtime();
    als_iter(tensor, factor, V, iter);
    iter_time = MPI_Wtime() - iter_start;

    MPI_Reduce(&iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    fitness = calc_fitness(factor);

    if (proc_id == 0) {
      cout << "Iteration " << iter << " max time: " << setw(width) << max_time;
      cout << " seconds; Fitness: " << fitness << "\n";
    }

    if (fitness == 1. || abs(fitness - prev_fitness) < tolerance)
      break;

    prev_fitness = fitness;
  }

}


void CoarseMPIALSSolver::als_iter(CoarseTensor &tensor, CoarseFactor &factor,
                                  Mat &V, int iter) {
  using namespace std;

  int proc_id = tensor.proc_id;
  uint64_t rank = factor.rank;
  int width = 8;

  double iter_start;
  double iter_time;
  double max_time;

  /*
   * Iteration A
   */
  V = (factor.BTB.cwiseProduct(factor.CTC).llt().solve(factor.ID));

  iter_start = MPI_Wtime();
  mttkrp_MA(tensor, factor, 0);
  factor.MA = factor.MA * V;
  normalize(factor, factor.MA, iter, 0);
  factor.local_ATA = factor.MA.transpose() * factor.MA;

  MPI_Allreduce(factor.local_ATA.data(), factor.ATA.data(), rank * rank,
             MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  // Allgatherv on MA
  MPI_Allgatherv(factor.MA.data(), tensor.counts[0][proc_id], MPI_DOUBLE,
                 factor.A.data(), &tensor.counts[0].front(),
                 &tensor.disps[0].front(), MPI_DOUBLE, MPI_COMM_WORLD);
  iter_time = MPI_Wtime() - iter_start;

  MPI_Reduce(&iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (proc_id == 0)
    cout << "A MTTKRP max time: " << setw(width) << max_time << " seconds;\n";

//  iter_start = MPI_Wtime();
//  factor.ATA = factor.A.transpose() * factor.A;
//  iter_time = MPI_Wtime() - iter_start;

//  MPI_Reduce(&iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//  if (proc_id == 0)
//    cout << "ATA max time: " << setw(width) << max_time << " seconds\n";

  /*
   * Iteration B
   */
  V = (factor.ATA.cwiseProduct(factor.CTC).llt().solve(factor.ID));

  iter_start = MPI_Wtime();
  mttkrp_MB(tensor, factor, 1);
  factor.MB = factor.MB * V;
  normalize(factor, factor.MB, iter, 1);
  factor.local_BTB = factor.MB.transpose() * factor.MB;

  MPI_Allreduce(factor.local_BTB.data(), factor.BTB.data(), rank * rank,
             MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  // Allgatherv on MB
  MPI_Allgatherv(factor.MB.data(), tensor.counts[1][proc_id], MPI_DOUBLE,
                 factor.B.data(), &tensor.counts[1].front(),
                 &tensor.disps[1].front(), MPI_DOUBLE, MPI_COMM_WORLD);
  iter_time = MPI_Wtime() - iter_start;

  MPI_Reduce(&iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (proc_id == 0)
    cout << "B MTTKRP max time: " << setw(width) << max_time << " seconds;\n";

//  iter_start = MPI_Wtime();
//  factor.BTB = factor.B.transpose() * factor.B;
//  iter_time = MPI_Wtime() - iter_start;

//  MPI_Reduce(&iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//  if (proc_id == 0)
//    cout << "BTB max time: " << setw(width) << max_time << " seconds\n";

  /*
   * Iteration C
   */
  V = (factor.ATA.cwiseProduct(factor.BTB).llt().solve(factor.ID));

  iter_start = MPI_Wtime();
  mttkrp_MC(tensor, factor, 2);

  // Allgatherv on MC_copy first, for fitness computation
  MPI_Allgatherv(factor.MC.data(), tensor.counts[2][proc_id], MPI_DOUBLE,
                 factor.MC_copy.data(), &tensor.counts[2].front(),
                 &tensor.disps[2].front(), MPI_DOUBLE, MPI_COMM_WORLD);

  factor.MC = factor.MC * V;
  normalize(factor, factor.MC, iter, 2);
  factor.local_CTC = factor.MC.transpose() * factor.MC;

  MPI_Allreduce(factor.local_CTC.data(), factor.CTC.data(), rank * rank,
                MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  // Allgatherv on MC
  MPI_Allgatherv(factor.MC.data(), tensor.counts[2][proc_id], MPI_DOUBLE,
                 factor.C.data(), &tensor.counts[2].front(),
                 &tensor.disps[2].front(), MPI_DOUBLE, MPI_COMM_WORLD);
  iter_time = MPI_Wtime() - iter_start;

  MPI_Reduce(&iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (proc_id == 0)
    cout << "C MTTKRP max time: " << setw(width) << max_time << " seconds;\n";

//  iter_start = MPI_Wtime();
//  factor.CTC = factor.C.transpose() * factor.C;
//  iter_time = MPI_Wtime() - iter_start;
//
//  MPI_Reduce(&iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//  if (proc_id == 0)
//    cout << "CTC max time: " << setw(width) << max_time << " seconds\n";
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


// Mat M: MA / MB / MC
void CoarseMPIALSSolver::normalize(CoarseFactor& factor, Mat& M, int iter,
                                   int mode) {
  using namespace std;
  uint64_t rank = factor.rank;

  double start_time;
  double iter_time;
  double max_time;

  start_time = MPI_Wtime();

  if (iter == 0) {   // L2 norm in the first iteration
    M.colwise().normalize();
  } else {           // Max norm for later iterations
    // Step 1: local lambda
    factor.local_lambda = M.colwise().maxCoeff();
    for (uint64_t r = 0; r < rank; r++)
      factor.local_lambda(r) = std::max(factor.local_lambda(r), 1.0);

    // Step 2: reduce to get global lambda
    MPI_Allreduce(factor.local_lambda.data(), factor.global_lambda.data(), rank,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    factor.lambda_inverse = (1 / factor.global_lambda.array()).matrix();
    M = M * factor.lambda_inverse.asDiagonal();
  }

  iter_time = MPI_Wtime() - start_time;

  MPI_Reduce(&iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (factor.proc_id == 0) {
    string tmp;
    if (mode == 0)
      tmp = "A";
    else if (mode == 1)
      tmp = "B";
    else
      tmp = "C";

    cout << tmp;
    cout << " normalization max time: " << setw(10) << max_time << " seconds; ";
  }
}


double CoarseMPIALSSolver::calc_kruskal_norm(CoarseFactor& factor) {
  Mat tmp = factor.ATA.cwiseProduct(factor.BTB).cwiseProduct(factor.CTC);
  Mat res = factor.global_lambda.transpose() * tmp * factor.global_lambda;
  return *res.data();
}

double CoarseMPIALSSolver::calc_kruskal_inner(CoarseFactor& factor) {
  Mat tmp = factor.MC_copy.cwiseProduct(factor.C);
  Mat res = factor.ones.transpose() * tmp * factor.global_lambda;
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



} // namespace litetensor
