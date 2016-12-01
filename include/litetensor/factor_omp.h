#ifndef LITETENSOR_FACTOR_OMP_H
#define LITETENSOR_FACTOR_OMP_H

#include <math.h>

#include <litetensor/config.h>
#include <litetensor/types.h>

namespace litetensor {

/*
 * Factor struct stores all factors and intermediate results
 * for single node shared address model
 */
struct Factor {
  uint64_t I, J, K;
  uint64_t rank;
  int num_threads;

  Mat A, B, C;         // Shapes: A = (I, R), B = (J, R), C = (K, R)
  Mat MA, MB, MC;
  Mat ATA, BTB, CTC;   // Shape: (R, R)
  Mat ID;              // Identity matrix

  Vec lambda;
  Vec ones;            // length = K

  // Parameters to track fitness
  double frob_norm;
  double frob_norm_sq;

  // Copy parameters from tensor, and allocate memory for buffer matrices
  Factor(RawTensor& tensor, Config& config) {
    using namespace Eigen;

    I = tensor.I;
    J = tensor.J;
    K = tensor.K;
    rank = config.rank;
    num_threads = config.num_threads;

    frob_norm = tensor.frob_norm;
    frob_norm_sq = sqrt(frob_norm);

    // Allocate memory
    A = MatrixXd::Random(I, rank);
    B = MatrixXd::Random(J, rank);
    C = MatrixXd::Random(K, rank);
    lambda = VectorXd(rank);

    ones = VectorXd(K).setOnes();

    ATA = MatrixXd(rank, rank);
    BTB = MatrixXd(rank, rank);
    CTC = MatrixXd(rank, rank);

    ID = MatrixXd(rank, rank).setIdentity();

    MA = MatrixXd(I, rank);
    MB = MatrixXd(J, rank);
    MC = MatrixXd(K, rank);
  }

};

}


#endif  // LITETENSOR_FACTOR_H
