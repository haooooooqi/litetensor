#include <iomanip>

#include <litetensor/tensor_mpi_coarse.h>

namespace litetensor {

/*
 * Methods of Partition class
 */
void Partitioner::print_tensor_stats() {
  using namespace std;

  cout << "=============== Tensor statistics ===============" << "\n";
  cout << "MPI processes: " << num_procs << "\n";
  cout << "Shape: " << I << "x" << J << "x" << K << "; ";
  cout << "Non-zeros: " << nnz << "\n";

  // Range of each process
  for (int i = 0; i < num_procs; i++) {
    cout << "Process " << setw(2) << i << " ";
    for (int m = 0; m < 3; m++) {
      cout << "Mode " << m;
      cout << " Local nnz: " << setw(9) << proc_nnz[m][i];
      cout << " Start: " << setw(7) << start_indices[m][i];
      cout << " End: " << setw(7) << end_indices[m][i] << "; ";
    }
    cout << "\n";
  }

}


Partitioner::Partitioner(Config& config) {
  using namespace std;

  FILE* fp = fopen(config.tensor_file.c_str(), "r");
  if (!fp)
    cout << "ERROR: can't open tensor file" << "\n";

  get_dim(fp);
  num_procs = config.num_procs;
  ave_nnz = nnz / config.num_procs;
  count_slice_nnz(fp);

  // Do partition on each mode
  partition_mode(0);
  partition_mode(1);
  partition_mode(2);

  // Check correctness in partition
  check_partition();
  print_tensor_stats();

  // Send partition info to each process


  // Close file
  int close = fclose(fp);
  if (close != 0)
    cout << "ERROR: can't close tensor file" << "\n";
}


void Partitioner::count_slice_nnz(FILE* fp) {
  using namespace std;

  // Allocate memory
  start_indices = vector<vector<uint64_t>>(3, vector<uint64_t>(num_procs));
  end_indices = vector<vector<uint64_t>>(3, vector<uint64_t>(num_procs));
  proc_nnz = vector<vector<uint64_t>>(3, vector<uint64_t>(num_procs));

  slice_nnz = vector<vector<uint64_t>>(3, vector<uint64_t>());
  slice_nnz[0] = vector<uint64_t>(I);
  slice_nnz[1] = vector<uint64_t>(J);
  slice_nnz[2] = vector<uint64_t>(K);

  // Count nnz in each slice
  rewind(fp);

  uint64_t i, j, k;
  char* line = NULL;
  ssize_t read;
  size_t len = 0;
  char* ptr = NULL;

  // Count nnz in each slice
  while ((read = getline(&line, &len, fp)) != -1) {
    // ptr = line;
    i = strtoull(line, &ptr, 10);
    ptr ++;
    j = strtoull(ptr, &ptr, 10);
    ptr ++;
    k = strtoull(ptr, &ptr, 10);
    ptr ++;

    i--; j--; k--;      // Convert to 0-based indices
    slice_nnz[0][i] ++;
    slice_nnz[1][j] ++;
    slice_nnz[2][k] ++;
  }
}


void Partitioner::partition_mode(int mode) {
  // Determine mode
  uint64_t dim = 0;
  if (mode == 0)
    dim = I;
  else if (mode == 1)
    dim = J;
  else if (mode == 2)
    dim = K;
  else
    std::cout << "ERROR: mode should be smaller than 3\n";

  uint64_t start_idx = 0;
  uint64_t cur_nnz = 0;
  int cur_proc = 0;

  for (uint64_t i = 0; i < dim && cur_proc < num_procs; i++) {
    cur_nnz += slice_nnz[mode][i];

    if (i + 1 < dim && cur_nnz + slice_nnz[mode][i+1] > ave_nnz &&
            cur_proc < num_procs - 1) {
      uint64_t gap1 = ave_nnz - cur_nnz;
      uint64_t gap2 = cur_nnz + slice_nnz[mode][i+1] - ave_nnz;

      start_indices[mode][cur_proc] = start_idx;
      if (gap1 <= gap2) {                        // i || i + 1
        end_indices[mode][cur_proc] = i + 1;
        proc_nnz[mode][cur_proc] = cur_nnz;
      } else {                                   // i, i + 1 || i + 2
        end_indices[mode][cur_proc] = i + 2;
        proc_nnz[mode][cur_proc] = cur_nnz + slice_nnz[mode][i+1];
        i ++;
      }

      start_idx = end_indices[mode][cur_proc];
      cur_nnz = 0;
      cur_proc ++;
    }
  }

  // Last process
  start_indices[mode][cur_proc] = start_idx;
  end_indices[mode][cur_proc] = dim;
  proc_nnz[mode][cur_proc] = cur_nnz;
}


bool Partitioner::check_partition() {
  using namespace std;

  // Check range consistency
  for (int mode = 0; mode < 3; mode ++) {
    uint64_t tmp_nnz = 0;

    for (int proc_id = 0; proc_id < num_procs; proc_id++) {
      // Check 1: consistency in start and end
      if (proc_id < num_procs - 1 &&
              start_indices[mode][proc_id + 1] != end_indices[mode][proc_id]) {
        cout << "ERROR: start and end index in-consistent in mode " << mode;
        cout << "\n";
        return false;
      }

      tmp_nnz += proc_nnz[mode][proc_id];
    }

    if (tmp_nnz != nnz) {
      cout << "ERROR: nnz in-consistent in mode " << mode << "\n";
      return false;
    }

  }


  return true;
}


void Partitioner::get_dim(FILE* fp) {
  I = 0; J = 0; K = 0;
  nnz = 0;

  rewind(fp);
  uint64_t i, j, k;
  char* line = NULL;
  ssize_t read;
  size_t len = 0;
  char* ptr = NULL;

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


/*
 * Methods of CoarseTensor class
 */



}   // litetensor
