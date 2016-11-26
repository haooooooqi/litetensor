#include <stdio.h>
#include <stdlib.h>

#include <litetensor/tensor.h>

namespace litetensor {

SequentialTensor::SequentialTensor(std::string tensor_file) {
  read_file_c(tensor_file);
  print_tensor_stats();
}

void SequentialTensor::read_file(std::string tensor_file) {
  using namespace std;

  ifstream infile;
  infile.open(tensor_file);

  if (!infile.is_open())
    cout << "ERROR: can't open tensor file" << "\n";

  get_dim(infile);
  allocate_tensor();

  infile.clear();
  infile.seekg(0);

  fill_tensor(infile);
  infile.close();
}

void SequentialTensor::read_file_c(std::string tensor_file) {
  using namespace std;

  FILE* fp = fopen(tensor_file.c_str(), "r");
  if (!fp)
    cout << "ERROR: can't open tensor file" << "\n";

  get_dim_c(fp);
  allocate_tensor();
  fill_tensor_c(fp);

  int close = fclose(fp);
  if (close != 0)
    cout << "ERROR: can't close tensor file" << "\n";
}


void SequentialTensor::print_tensor_stats() {
  using namespace std;

  cout << "=============== Tensor statistics ===============" << "\n";
  cout << "Shape: " << I_ << "x" << J_ << "x" << K_ << "; ";
  cout << "Non-zeros: " << nnz_ << "\n";
}


void SequentialTensor::allocate_tensor() {
  using namespace std;

  indices_ = vector<vector<vector<uint64_t>>>(3, vector<vector<uint64_t>>());
  indices_[0] = vector<vector<uint64_t>>(I_, vector<uint64_t>());
  indices_[1] = vector<vector<uint64_t>>(J_, vector<uint64_t>());
  indices_[2] = vector<vector<uint64_t>>(K_, vector<uint64_t>());

  vals_ = vector<vector<vector<double>>>(3, vector<vector<double>>());
  vals_[0] = vector<vector<double>>(I_, vector<double>());
  vals_[1] = vector<vector<double>>(J_, vector<double>());
  vals_[2] = vector<vector<double>>(K_, vector<double>());
}


void SequentialTensor::fill_tensor(std::ifstream& infile) {
  using namespace std;

  // Calculate frobenius norm
  frob_norm_ = 0.0;

  if (!infile.is_open())
    cout << "ERROR: can't open tensor file" << "\n";

  uint64_t i, j, k;
  double val;
  char c;
  while ((infile >> i >> c >> j >> c >> k >> c >> val) && (c == ',')) {
    i--; j--; k--;

    frob_norm_ += val * val;

    // Push indices and values
    indices_[0][i].push_back(k * J_ + j);
    indices_[1][j].push_back(k * I_ + i);
    indices_[2][k].push_back(j * I_ + i);
    vals_[0][i].push_back(val);
    vals_[1][j].push_back(val);
    vals_[2][k].push_back(val);
  }
}

void SequentialTensor::fill_tensor_c(FILE* fp) {
  uint64_t i, j, k;
  double val;
  char* line = NULL;
  ssize_t read;
  size_t len = 0;
  char* ptr = NULL;

  rewind(fp);     // Point to file head
  while ((read = getline(&line, &len, fp)) != -1) {
    ptr = line;
    i = strtoull(ptr, &ptr, 10) - 1;
    ptr ++;
    j = strtoull(ptr, &ptr, 10) - 1;
    ptr ++;
    k = strtoull(ptr, &ptr, 10) - 1;
    ptr ++;
    val = strtod(ptr, &ptr);

    frob_norm_ += val * val;
    // Push indices and values
    indices_[0][i].push_back(k * J_ + j);
    indices_[1][j].push_back(k * I_ + i);
    indices_[2][k].push_back(j * I_ + i);
    vals_[0][i].push_back(val);
    vals_[1][j].push_back(val);
    vals_[2][k].push_back(val);
  }
}

void SequentialTensor::get_dim(std::ifstream& infile) {
  using namespace std;

  nnz_ = 0; I_ = 0; J_ = 0; K_ = 0;

  uint64_t i, j, k;
  double val;
  char c;
  while ((infile >> i >> c >> j >> c >> k >> c >> val) && (c == ',')) {
    I_ = max(I_, i);
    J_ = max(J_, j);
    K_ = max(K_, k);
    nnz_ ++;
  }
}

void SequentialTensor::get_dim_c(FILE* fp) {
  nnz_ = 0; I_ = 0; J_ = 0; K_ = 0;

  uint64_t i, j, k;
  char* line = NULL;
  ssize_t read;
  size_t len = 0;
  char* ptr = NULL;

/*
  if ((read = getline(&line, &len, fp)) != -1) {
      I_ = strtoull(line, &ptr, 10);
      ptr ++;
      J_ = strtoull(ptr, &ptr, 10);
      ptr ++;
      K_ = strtoull(ptr, &ptr, 10);
  }
  */
  rewind(fp);     // Point to file head
  while ((read = getline(&line, &len, fp)) != -1) {
    // ptr = line;
    i = strtoull(line, &ptr, 10);
    ptr ++;
    j = strtoull(ptr, &ptr, 10);
    ptr ++;
    k = strtoull(ptr, &ptr, 10);
    ptr ++;

    I_ = std::max(I_, i);
    J_ = std::max(J_, j);
    K_ = std::max(K_, k);
    nnz_ ++;
  }
}

}
