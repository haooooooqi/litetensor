#ifndef LITETENSOR_TENSOR_MPI_COARSE_H
#define LITETENSOR_TENSOR_MPI_COARSE_H

#include <mpi.h>

#include <litetensor/config.h>
#include <litetensor/tensor.h>

namespace litetensor {

class Partitioner {
private:
  // Number of elements in each slice
  std::vector<std::vector<uint64_t>> slice_nnz;      // shape = I, J, K

  // Get tensor shape
  void get_dim(FILE* fp);

  // Count nnz in each slice
  void count_slice_nnz(FILE* fp);

  // Do partition on each mode
  void partition_mode(int mode);

  // More fine-grained partition
  void partition_mode_fine(int mode);

  // TODO: Optimal partition
  void partition_mode_optimal(int mode) {}

  // Print tensor statistics
  void print_tensor_stats();

  // Check consistency of partition
  bool check_partition();

public:
  uint64_t I, J, K;
  uint64_t nnz;
  int num_procs;

  // Range style: [start, end)
  std::vector<std::vector<uint64_t>> start_indices;  // shape = 3 * num_procs
  std::vector<std::vector<uint64_t>> end_indices;    // shape = 3 * num_procs
  std::vector<std::vector<uint64_t>> proc_nnz;       // shape = 3 * num_procs

  /*
   * Partition the tensor in each node.
   */
  void partition(Config& config);

};


/*
 * Tensor struct used in coarse-grained MPI code
 */
struct CoarseTensor {
  int proc_id;
  int num_procs;
  uint64_t I, J, K;

  double frob_norm;
  double frob_norm_sq;

  // 0-based index
  std::vector<uint64_t> start_rows, end_rows, num_rows;

  // Tensor value and indices
  std::vector<std::vector<std::vector<uint64_t>>> indices;   // Indices
  std::vector<std::vector<std::vector<double>>> vals;        // Values

  // Element counts and displacements array
  std::vector<std::vector<int>> counts;     // shape = 3 * num_procs
  std::vector<std::vector<int>> disps;      // shape = 3 * num_procs

  // Construct tensor
  void construct_tensor(Partitioner& partitioner, Config& config);

  // Scatter information
  void scatter_partition(Partitioner& partitioner, Config& config);

  // Read file and fill tensor
  void fill_tensor(Config& config);

};



}   // namespace litetensor



#endif //LITETENSOR_TENSOR_MPI_COARSE_H
