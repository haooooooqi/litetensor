#ifndef LITETENSOR_ALS_SEQUENTIAL_H
#define LITETENSOR_ALS_SEQUENTIAL_H

#include <litetensor/tensor.h>
#include <litetensor/config.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Cholesky>

namespace litetensor {

typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;

class SequentialALSSolver {
public:
  uint64_t I_, J_, K_;
  uint64_t JK_, IK_, IJ_;
  uint64_t rank_;

  Mat A_, B_, C_;         // Shapes: A_ = (I, R), B_ = (J, R), C_ = (K, R)
  Mat MA_, MB_, MC_;
  Mat ATA_, BTB_, CTC_;   // Shape: (R, R)
  Mat ID_;                // Identity matrix

  Vec lambda_;
  Vec ones_;              // length = K

  // Parameters to track fitness
  double frob_norm_;
  double frob_norm_sq_;

  // Copy parameters from tensor
  void copy_params(SequentialTensor& tensor);

  /*
   * Conduct matricized tensor times Khatri-Rao product (MTTKRP)
   * For mode 1
   */
  void mttkrp_MA(SequentialTensor& tensor, Mat& MA, Mat& C, Mat& B,
                 uint64_t mode);

  // MTTKRP for mode 2
  void mttkrp_MB(SequentialTensor& tensor, Mat& MB, Mat& C, Mat& A,
                 uint64_t mode);

  // MTTKRP for mode 3
  void mttkrp_MC(SequentialTensor& tensor, Mat& MC, Mat& B, Mat& A,
                 uint64_t mode);

  // Normalize factor matrix
  void normalize(Mat& M, int iter);

  // Alternating Least Square algorithm (ALS)
  void als(SequentialTensor& tensor, int max_iters, double tolerance);

  void als_iter(SequentialTensor& tensor, Mat& V, int iter);

  // Calculate fitness
  double calc_fitness(Mat& ATA, Mat& BTB, Mat& CTC, Mat& MC, Mat& C);

  // Calculate Kruskal norm
  double calc_kruskal_norm(Mat& ATA, Mat& BTB, Mat& CTC);

  // Calculate Kruskal inner product
  double calc_kruskal_inner(Mat& MC, Mat& C);

  // Check decomposition result
  void check_correctness(SequentialTensor& tensor);

  // Do decomposition
  void decompose(SequentialTensor& tensor, int max_iter, double tolerance);

  // Constructor
  SequentialALSSolver(SequentialTensor& tensor, uint64_t rank);


};

}

#endif //LITETENSOR_ALS_SEQUENTIAL_H
