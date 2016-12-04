#ifndef LITETENSOR_FACTOR_MPI_COARSE_H
#define LITETENSOR_FACTOR_MPI_COARSE_H

#include <litetensor/types.h>

namespace litetensor {

/*
 * CoarseFactor - Factor designed for coarse grained MPI code
 */
struct CoarseFactor {
  uint64_t I, J, K;
  uint64_t rank;

  Mat A, B, C;

  // A copy of C, used in fitness calculation
  Mat MC_copy;

  // Result of MTTKRP, shape = (partial rows, rank)
  // Matrices to be communicated
  Mat MA, MB, MC;

  Mat ATA, BTB, CTC;   // Shape: (R, R)
  Mat ID;              // Identity matrix

  Vec lambda;
  Vec ones;

  double frob_norm;
  double frob_norm_sq;

  CoarseFactor(CoarseTensor& tensor, Config& config) {
    using namespace Eigen;

    frob_norm = tensor.frob_norm;
    frob_norm_sq = tensor.frob_norm_sq;

    I = tensor.I;
    J = tensor.J;
    K = tensor.K;
    rank = config.rank;

    // Allocate dense matrices, each node initialized its own A, B, C
    A = MatrixXd::Random(I, rank);
    B = MatrixXd::Random(J, rank);
    C = MatrixXd::Random(K, rank);

    lambda = VectorXd(rank);
    ones = VectorXd(K).setOnes();

    MC_copy = MatrixXd(K, rank);

    ATA = MatrixXd(rank, rank);
    BTB = MatrixXd(rank, rank);
    CTC = MatrixXd(rank, rank);

    ID = MatrixXd(rank, rank).setIdentity();

    // MTTKRP matrices
    MA = MatrixXd(tensor.num_rows[0], rank);
    MB = MatrixXd(tensor.num_rows[1], rank);
    MC = MatrixXd(tensor.num_rows[2], rank);

  }

};



}


#endif //LITETENSOR_FACTOR_MPI_COARSE_H
