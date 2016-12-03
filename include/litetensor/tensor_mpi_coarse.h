#ifndef LITETENSOR_TENSOR_MPI_COARSE_H
#define LITETENSOR_TENSOR_MPI_COARSE_H

#include <mpi.h>

#include <litetensor/config.h>
#include <litetensor/tensor.h>

namespace litetensor {

class Partitioner {
private:
  uint64_t I, J, K;
  uint64_t nnz;
  int num_procs;
  uint64_t ave_nnz;      // nnz in each process

  // Range style: [start, end)
  std::vector<std::vector<uint64_t>> start_indices;  // shape = 3 * num_procs
  std::vector<std::vector<uint64_t>> end_indices;    // shape = 3 * num_procs
  std::vector<std::vector<uint64_t>> proc_nnz;       // shape = 3 * num_procs

  // Number of elements in each slice
  std::vector<std::vector<uint64_t>> slice_nnz;      // shape = I, J, K

  // Get tensor shape
  void get_dim(FILE* fp);

  // Count nnz in each slice
  void count_slice_nnz(FILE* fp);

  // Do partition on each mode
  void partition_mode(int mode);

  // Print tensor statistics
  void print_tensor_stats();

  // Check consistency of partition
  bool check_partition();

public:
  /*
   * Partition the tensor in each node.
   * Send start and end index to each process
   */
  Partitioner(Config& config);

  // Broadcast partition results to other processes

};


struct CoarseTensor {
  uint64_t I, J, K;

  // Constructor, get row info, only store a range of rows
  CoarseTensor(Config& config, int proc_id) {}

};




}



#endif //LITETENSOR_TENSOR_MPI_COARSE_H
