#ifndef LITETENSOR_FACTOR_MPI_COARSE_H
#define LITETENSOR_FACTOR_MPI_COARSE_H

#include <litetensor/types.h>

namespace litetensor {

/*
 * CoarseFactor - Factor designed for coarse grained MPI code
 */
struct CoarseFactor {
  int proc_id;
  uint64_t I, J, K;
  uint64_t rank;

  Mat A, B, C;

  // A copy of C, used in fitness calculation
  Mat MC_copy;

  // Result of MTTKRP, shape = (partial rows, rank)
  // Matrices to be communicated
  Mat MA, MB, MC;

  Mat ATA, BTB, CTC;   // Shape: (R, R)
  Mat local_ATA, local_BTB, local_CTC;   // Shape: (R, R)
  Mat ID;              // Identity matrix

  Vec local_lambda;
  Vec global_lambda;
  Vec lambda_inverse;
  Vec ones;

  double frob_norm;
  double frob_norm_sq;

  CoarseFactor(CoarseTensor& tensor, Config& config) {
    using namespace Eigen;

    proc_id = tensor.proc_id;
    frob_norm = tensor.frob_norm;
    frob_norm_sq = tensor.frob_norm_sq;

    I = tensor.I;
    J = tensor.J;
    K = tensor.K;
    rank = config.rank;

    // MTTKRP matrices
    MA = MatrixXd::Random(tensor.num_rows[0], rank);
    MB = MatrixXd::Random(tensor.num_rows[1], rank);
    MC = MatrixXd::Random(tensor.num_rows[2], rank);

    A = MatrixXd(I, rank);
    B = MatrixXd(J, rank);
    C = MatrixXd(K, rank);

    // Allgatherv
    MPI_Allgatherv(MA.data(), tensor.counts[0][proc_id], MPI_DOUBLE,
                   A.data(), &tensor.counts[0][0],
                   &tensor.disps[0][0], MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Allgatherv(MB.data(), tensor.counts[1][proc_id], MPI_DOUBLE,
                   B.data(), &tensor.counts[1][0],
                   &tensor.disps[1][0], MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Allgatherv(MC.data(), tensor.counts[2][proc_id], MPI_DOUBLE,
                   C.data(), &tensor.counts[2][0],
                   &tensor.disps[2][0], MPI_DOUBLE, MPI_COMM_WORLD);


    local_lambda = VectorXd(rank);
    global_lambda = VectorXd(rank);
    lambda_inverse = VectorXd(rank);

    ones = VectorXd(K).setOnes();

    MC_copy = MatrixXd(K, rank);

    ATA = MatrixXd(rank, rank);
    BTB = MatrixXd(rank, rank);
    CTC = MatrixXd(rank, rank);

    local_ATA = MatrixXd(rank, rank);
    local_BTB = MatrixXd(rank, rank);
    local_CTC = MatrixXd(rank, rank);

    ID = MatrixXd(rank, rank).setIdentity();
  }

};


} // namespace litetensor


#endif //LITETENSOR_FACTOR_MPI_COARSE_H
