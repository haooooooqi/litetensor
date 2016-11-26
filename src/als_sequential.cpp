#include <math.h>

#include <chrono>

#include <litetensor/als_sequential.h>


namespace litetensor {

void SequentialALSSolver::copy_params(SequentialTensor& tensor) {
  frob_norm_ = tensor.frob_norm_;
  frob_norm_sq_ = sqrt(frob_norm_);

  I_ = tensor.I_;
  J_ = tensor.J_;
  K_ = tensor.K_;
  JK_ = J_ * K_;
  IK_ = I_ * K_;
  IJ_ = I_ * K_;
}

void SequentialALSSolver::mttkrp_MA(SequentialTensor& tensor, Mat& MA, Mat& C,
                                    Mat& B, uint64_t mode) {
    // Initialize MA to 0s, very important
  MA.setZero();
  for (uint64_t i = 0; i < I_; i++) {   // Each row of MA(i, :)
    for (uint64_t idx = 0; idx < tensor.indices_[mode][i].size(); idx++) {
      uint64_t j = tensor.indices_[mode][i][idx] % J_;
      uint64_t k = tensor.indices_[mode][i][idx] / J_;
      MA.row(i) += tensor.vals_[mode][i][idx] *
              (B.row(j).cwiseProduct(C.row(k)));
    }
  }
}

  // MTTKRP for mode 2
void SequentialALSSolver::mttkrp_MB(SequentialTensor& tensor,
                                    Mat& MB, Mat& C, Mat& A, uint64_t mode) {

  MB.setZero();
  for (uint64_t j = 0; j < J_; j++) {
    for (uint64_t idx = 0; idx < tensor.indices_[mode][j].size(); idx++) {
      uint64_t i = tensor.indices_[mode][j][idx] % I_;
      uint64_t k = tensor.indices_[mode][j][idx] / I_;
      MB.row(j) += tensor.vals_[mode][j][idx] *
              (C.row(k).cwiseProduct(A.row(i)));
    }
}
}

// MTTKRP for mode 3
void SequentialALSSolver::mttkrp_MC(SequentialTensor& tensor, Mat& MC, Mat& B, Mat& A, uint64_t mode) {
  MC.setZero();
  for (uint64_t k = 0; k < K_; k++) {
    for (uint64_t idx = 0; idx < tensor.indices_[mode][k].size(); idx++) {
      uint64_t i = tensor.indices_[mode][k][idx] % I_;
      uint64_t j = tensor.indices_[mode][k][idx] / I_;
      MC.row(k) += tensor.vals_[mode][k][idx] *
              (B.row(j).cwiseProduct(A.row(i)));
    }
  }
}

void SequentialALSSolver::normalize(Mat& M, int iter) {
  if (iter == 0) {   // L2 norm in the first iteration
    for (uint64_t r = 0; r < rank_; r++)
      M.col(r).normalize();
  } else {           // Max norm for later iterations
    for (uint64_t r = 0; r < rank_; r++) {
      lambda_(r) = std::max(M.col(r).maxCoeff(), 1.0);
      M.col(r) /= lambda_(r);
    }
  }
}


void SequentialALSSolver::als_iter(SequentialTensor& tensor, Mat& V, int iter) {
  // Update A
  V = (BTB_.cwiseProduct(CTC_).llt().solve(ID_));
  mttkrp_MA(tensor, MA_, C_, B_, 0);
  A_ = MA_ * V;
  normalize(A_, iter);
  ATA_ = A_.transpose() * A_;

  // Update B
  V = (ATA_.cwiseProduct(CTC_).llt().solve(ID_));
  mttkrp_MB(tensor, MB_, C_, A_, 1);
  B_ = MB_ * V;
  normalize(B_, iter);
  BTB_ = B_.transpose() * B_;

  // Update C
  V = (ATA_.cwiseProduct(BTB_).llt().solve(ID_));
  mttkrp_MC(tensor, MC_, B_, A_, 2);
  C_ = MC_ * V;
  normalize(C_, iter);
  CTC_ = C_.transpose() * C_;
}


void SequentialALSSolver::als(SequentialTensor& tensor, int max_iters,
                              double tolerance) {
  using namespace Eigen;
  using namespace std;
  using namespace std::chrono;
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> dsec;

  cout << "=============== Decomposing Tensor ==============" << endl;
  cout << "Max iterations: " << max_iters << "; " <<  "Rank: " << rank_ << "; ";
  cout << "Tolerance: " << tolerance << endl;

  ATA_ = A_.transpose() * A_;
  BTB_ = B_.transpose() * B_;
  CTC_ = C_.transpose() * C_;

  Mat V = MatrixXd(rank_, rank_);

  double prev_fitness = 1;
  double fitness = 0;
  high_resolution_clock::time_point iter_start;
  double iter_time;

  for (int iter = 0; iter < max_iters; iter++) {
    iter_start = Clock::now();
    als_iter(tensor, V, iter);
    iter_time = duration_cast<dsec>(Clock::now() - iter_start).count();

    // Check fitness
    fitness = calc_fitness(ATA_, BTB_, CTC_, MC_, C_);
    cout << "Time: " << iter_time << ", ";
    cout << "Iteration: " << iter + 1 << ", Fitness: " << fitness << endl;

    if (fitness == 1. || abs(fitness - prev_fitness) < tolerance)
      break;

    prev_fitness = fitness;
  }


//  cout << A_ << B_ << C_ << lambda_ << endl;
}


void SequentialALSSolver::decompose(SequentialTensor& tensor, int max_iter
        , double tolerance) {
  als(tensor, max_iter, tolerance);
}


void SequentialALSSolver::check_correctness(SequentialTensor& tensor) {
  using namespace Eigen;
  using namespace std;

  vector<vector<vector<double>>> outer_product(I_, vector<vector<double>>
            (J_, vector<double>(K_, 0.0)));

  for (uint64_t i = 0; i < I_; i++) {
    for (uint64_t j = 0; j < J_; j++) {
      for (uint64_t k = 0; k < K_; k++) {
        for (uint64_t r = 0; r < rank_; r++)
          outer_product[i][j][k] += lambda_(r)*A_(i, r)*B_(j, r)*C_(k, r);
      }
    }
  }

  // Compute difference between original tensor and decomposition results
  double diff = 0.0;
  for (uint64_t i = 0; i < I_; i++) {
    for (uint64_t idx = 0; idx < tensor.indices_[0][i].size(); idx++) {
      uint64_t j = tensor.indices_[0][i][idx] % J_;
      uint64_t k = tensor.indices_[0][i][idx] / J_;
      diff += abs(outer_product[i][j][k] - tensor.vals_[0][i][idx])
              / abs(tensor.vals_[0][i][idx]);
    }
  }

  cout << "Average difference: " << diff / tensor.nnz_ << endl;
}


SequentialALSSolver::SequentialALSSolver(SequentialTensor &tensor,
                                         uint64_t rank){
  using namespace std;
  using namespace Eigen;

  copy_params(tensor);

  // Allocate space
  A_ = MatrixXd::Random(I_, rank_);
  B_ = MatrixXd::Random(J_, rank_);
  C_ = MatrixXd::Random(K_, rank_);
  lambda_ = VectorXd(rank_);

  ones_ = VectorXd(K_).setOnes();

  ATA_ = MatrixXd(rank_, rank_);
  BTB_ = MatrixXd(rank_, rank_);
  CTC_ = MatrixXd(rank_, rank_);

  ID_ = MatrixXd(rank_, rank_).setIdentity();

  MA_ = MatrixXd(I_, rank);
  MB_ = MatrixXd(J_, rank);
  MC_ = MatrixXd(K_, rank);
}


double SequentialALSSolver::calc_kruskal_norm(Mat& ATA, Mat& BTB, Mat& CTC) {
  Mat tmp = ATA.cwiseProduct(BTB).cwiseProduct(CTC);
  Mat res = lambda_.transpose() * tmp * lambda_;
  return *res.data();
}

double SequentialALSSolver::calc_kruskal_inner(Mat& MC, Mat& C) {
  Mat tmp = MC.cwiseProduct(C);
  Mat res = ones_.transpose() * tmp * lambda_;
  return *res.data();
}


double SequentialALSSolver::calc_fitness(Mat& ATA, Mat& BTB, Mat& CTC,
                                         Mat& MC, Mat& C) {
  double kruskal_norm = calc_kruskal_norm(ATA, BTB, CTC);
  double kruskal_inner = calc_kruskal_inner(MC, C);

  double residual = frob_norm_ + kruskal_norm - (2 * kruskal_inner);

  if (residual > 0.0)
    residual = sqrt(residual);

  return 1 - (residual / frob_norm_sq_);
}

}
