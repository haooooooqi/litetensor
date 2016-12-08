#ifndef LITETENSOR_ALS_MPI_COARSE_H
#define LITETENSOR_ALS_MPI_COARSE_H

#include <litetensor/tensor_mpi_coarse.h>
#include <litetensor/factor_mpi_coarse.h>


namespace litetensor {

class CoarseMPIALSSolver {
public:
  void solve(CoarseTensor& tensor, Config& config);

private:
  void als(CoarseTensor& tensor, CoarseFactor& factor, Config& config);

  void als_iter(CoarseTensor& tensor, CoarseFactor& factor, Mat& V, int iter);

  // MTTKRP for mode 1
  void mttkrp_MA(CoarseTensor& tensor, CoarseFactor& factor, uint64_t mode);

  // MTTKRP for mode 2
  void mttkrp_MB(CoarseTensor& tensor, CoarseFactor& factor, uint64_t mode);

  // MTTKRP for mode 3
  void mttkrp_MC(CoarseTensor& tensor, CoarseFactor& factor, uint64_t mode);

  // Normalize factor matrix
  void normalize(CoarseFactor& factor, Mat& M, int iter, int mode);

  // Check fitness
  double calc_fitness(CoarseFactor& factor);

  // Calculate Kruskal norm, outer product
  double calc_kruskal_norm(CoarseFactor& factor);

  // Calculate Kruskal inner product
  double calc_kruskal_inner(CoarseFactor& factor);

};




}


#endif //LITETENSOR_ALS_MPI_COARSE_H
