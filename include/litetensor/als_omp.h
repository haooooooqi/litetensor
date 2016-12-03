#ifndef LITETENSOR_ALS_OMP_H
#define LITETENSOR_ALS_OMP_H

#include <litetensor/tensor.h>
#include <litetensor/types.h>
#include <litetensor/factor_omp.h>

namespace litetensor {

class OMPALSSolver {
private:

  /*
   * Conduct matricized tensor times Khatri-Rao product (MTTKRP)
   * For mode 1
   */
  void mttkrp_MA(RawTensor& tensor, Factor& factor, uint64_t mode);

  // MTTKRP for mode 2
  void mttkrp_MB(RawTensor& tensor, Factor& factor, uint64_t mode);

  // MTTKRP for mode 3
  void mttkrp_MC(RawTensor& tensor, Factor& factor, uint64_t mode);

  // Normalize factor matrix
  void normalize(Factor& factor, Mat& M, int iter);

  // Alternating Least Square algorithm (ALS)
  void als(RawTensor& tensor, Factor& factor, Config& config);

  void als_iter(RawTensor& tensor, Factor& factor, Mat& V, int iter);

  // Calculate fitness
  double calc_fitness(Factor& factor);

  // Calculate Kruskal norm
  double calc_kruskal_norm(Factor& factor);

  // Calculate Kruskal inner product
  double calc_kruskal_inner(Factor& factor);

public:
  // Do decomposition
  void decompose(RawTensor& tensor, Config& config);

};

}  // litetensor


#endif //LITETENSOR_ALS_OMP_H
