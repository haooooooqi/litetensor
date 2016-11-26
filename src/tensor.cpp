#include <litetensor/tensor.h>

namespace litetensor {

SequentialTensor::SequentialTensor(std::string tensor_file) {
  using namespace std;

  ifstream infile;
  infile.open(tensor_file);

  if (!infile.is_open())
    cout << "ERROR: can't open tensor file" << endl;

  get_dim(infile);
  allocate_tensor();

  infile.clear();
  infile.seekg(0);

  fill_tensor(infile);

  infile.close();

  // Show tensor statistics
  print_tensor_stats();
}


void SequentialTensor::print_tensor_stats() {
  using namespace std;

  cout << "=============== Tensor statistics ===============" << endl;
  cout << "Shape: " << I_ << "x" << J_ << "x" << K_ << "; ";
  cout << "Non-zeros: " << nnz_ << endl << endl;
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
    cout << "ERROR: can't open tensor file" << endl;

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


}
