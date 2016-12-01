#include <stdio.h>
#include <stdlib.h>

#include <litetensor/tensor.h>

namespace litetensor {

RawTensor::RawTensor(std::string tensor_file) {
  read_file_c(tensor_file);
  print_tensor_stats();
}

void RawTensor::read_file(std::string tensor_file) {
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

void RawTensor::read_file_c(std::string tensor_file) {
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


void RawTensor::print_tensor_stats() {
  using namespace std;

  cout << "=============== Tensor statistics ===============" << "\n";
  cout << "Shape: " << I << "x" << J << "x" << K << "; ";
  cout << "Non-zeros: " << nnz << "\n";
}


void RawTensor::allocate_tensor() {
  using namespace std;

  indices = vector<vector<vector<uint64_t>>>(3, vector<vector<uint64_t>>());
  indices[0] = vector<vector<uint64_t>>(I, vector<uint64_t>());
  indices[1] = vector<vector<uint64_t>>(J, vector<uint64_t>());
  indices[2] = vector<vector<uint64_t>>(K, vector<uint64_t>());

  vals = vector<vector<vector<double>>>(3, vector<vector<double>>());
  vals[0] = vector<vector<double>>(I, vector<double>());
  vals[1] = vector<vector<double>>(J, vector<double>());
  vals[2] = vector<vector<double>>(K, vector<double>());
}


void RawTensor::fill_tensor(std::ifstream& infile) {
  using namespace std;

  // Calculate frobenius norm
  frob_norm = 0.0;

  if (!infile.is_open())
    cout << "ERROR: can't open tensor file" << "\n";

  uint64_t i, j, k;
  double val;
  char c;
  while ((infile >> i >> c >> j >> c >> k >> c >> val) && (c == ',')) {
    i--; j--; k--;

    frob_norm += val * val;

    // Push indices and values
    indices[0][i].push_back(k * J + j);
    indices[1][j].push_back(k * I + i);
    indices[2][k].push_back(j * I + i);
    vals[0][i].push_back(val);
    vals[1][j].push_back(val);
    vals[2][k].push_back(val);
  }
}

void RawTensor::fill_tensor_c(FILE* fp) {
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

    frob_norm += val * val;
    // Push indices and values
    indices[0][i].push_back(k * J + j);
    indices[1][j].push_back(k * I + i);
    indices[2][k].push_back(j * I + i);
    vals[0][i].push_back(val);
    vals[1][j].push_back(val);
    vals[2][k].push_back(val);
  }
}

void RawTensor::get_dim(std::ifstream& infile) {
  using namespace std;

  nnz = 0; I = 0; J = 0; K = 0;

  uint64_t i, j, k;
  double val;
  char c;
  while ((infile >> i >> c >> j >> c >> k >> c >> val) && (c == ',')) {
    I = max(I, i);
    J = max(J, j);
    K = max(K, k);
    nnz ++;
  }
}

void RawTensor::get_dim_c(FILE* fp) {
  nnz = 0; I = 0; J = 0; K = 0;

  uint64_t i, j, k;
  char* line = NULL;
  ssize_t read;
  size_t len = 0;
  char* ptr = NULL;

/*
  if ((read = getline(&line, &len, fp)) != -1) {
      I = strtoull(line, &ptr, 10);
      ptr ++;
      J = strtoull(ptr, &ptr, 10);
      ptr ++;
      K = strtoull(ptr, &ptr, 10);
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

    I = std::max(I, i);
    J = std::max(J, j);
    K = std::max(K, k);
    nnz ++;
  }
}

}
