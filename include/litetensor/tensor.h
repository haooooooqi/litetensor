#ifndef LITETENSOR_TENSOR_H
#define LITETENSOR_TENSOR_H

#include <iostream>
#include <fstream>
#include <vector>

namespace litetensor {

/*
 * Tensor format for sequential ALS. Store slices for each mode.
 * For each slice, we store the indices and vals
 */
class SequentialTensor {
public:
  uint64_t I_, J_, K_;
  uint64_t nnz_;
  std::vector<std::vector<std::vector<uint64_t>>> indices_;   // Indices
  std::vector<std::vector<std::vector<double>>> vals_;     // Values

  // Frobenius norm
  double frob_norm_;

  SequentialTensor(std::string tensor_file);

private:
  // Print out tensor statistics
  void print_tensor_stats();

  // Read the file
  void read_file(std::string tensor_file);
  void read_file_c(std::string tensor_file);

  // Traverse the file and get the dimension
  void get_dim(std::ifstream& infile);
  void get_dim_c(FILE* fp);

  // Allocate space for tensor
  void allocate_tensor();

  // Fill the tensor
  void fill_tensor(std::ifstream& infile);
  void fill_tensor_c(FILE* fp);

};


}


#endif //LITETENSOR_TENSOR_H
