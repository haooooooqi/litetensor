#include <mpi.h>
#include <math.h>

#include <iomanip>

#include <litetensor/tensor_mpi_coarse.h>

namespace litetensor {

/*
 * Methods of Partition class
 */
void Partitioner::print_tensor_stats() {
  using namespace std;

  cout << "======================= Tensor statistics ======================\n";
  cout << "MPI processes: " << num_procs << "\n";
  cout << "Shape: " << I << "x" << J << "x" << K << "; ";
  cout << "Non-zeros: " << nnz << "\n";

  // Range of each process
  for (int i = 0; i < num_procs; i++) {
    cout << "Process " << setw(2) << i << "\n";
    for (int m = 0; m < 3; m++) {
      cout << "Mode " << m;
      cout << " Local nnz: " << setw(9) << proc_nnz[m][i];
      cout << " Start: " << setw(7) << start_indices[m][i];
      cout << " End: " << setw(7) << end_indices[m][i];
      cout << " Rows: " << setw(7) << end_indices[m][i] - start_indices[m][i];
      cout << ";\n";
    }
  }

}


void Partitioner::partition(Config& config) {
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
void CoarseTensor::scatter_partition(Partitioner &partitioner, Config &config) {
  using namespace std;

  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
  num_procs = config.num_procs;

  if (proc_id == 0) { // Prepare for parameters
    I = partitioner.I;
    J = partitioner.J;
    K = partitioner.K;
  }

  // Scatter shape info
  MPI_Bcast(&I, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&J, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&K, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

  /*
  cout << "From process " << proc_id << " Received I is " << I << "\n";
  cout << "From process " << proc_id << " Received J is " << J << "\n";
  cout << "From process " << proc_id << " Received K is " << K << "\n";
   */

  // Scatter row ranges
  start_rows = vector<uint64_t>(3);
  end_rows = vector<uint64_t>(3);
  num_rows = vector<uint64_t>(3);

  uint64_t* global_start_ptr = NULL;
  uint64_t* global_end_ptr = NULL;

  for (int m = 0; m < 3; m++) {
    if (proc_id == 0) {
      global_start_ptr = &partitioner.start_indices[m][0];
      global_end_ptr = &partitioner.end_indices[m][0];
    }

    MPI_Scatter(global_start_ptr, 1, MPI_INT64_T, &start_rows[m],
                1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Scatter(global_end_ptr, 1, MPI_INT64_T, &end_rows[m],
                1, MPI_INT64_T, 0, MPI_COMM_WORLD);

    num_rows[m] = end_rows[m] - start_rows[m];

    /*
    cout << "From process " << proc_id << " Mode " << m;
    cout << " Start: " << start_rows[m] << " End: " << end_rows[m];
    cout << " Number of rows: " << num_rows[m];
    cout << "\n";
     */
  }

  // Scatter counts and displacements
  counts = vector<vector<int>> (3, vector<int>(num_procs));
  disps = vector<vector<int>> (3, vector<int>(num_procs));

  if (proc_id == 0) {
    for (int m = 0; m < 3; m++) {
      for (int i = 0; i < num_procs; i++) {
        counts[m][i] = (int) config.rank * (partitioner.end_indices[m][i] -
                partitioner.start_indices[m][i]);
        disps[m][i] = (int) config.rank * partitioner.start_indices[m][i];
      }
    }
  }

  for (int m = 0; m < 3; m++) {
    MPI_Bcast(&counts[m][0], num_procs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&disps[m][0], num_procs, MPI_INT, 0, MPI_COMM_WORLD);
  }

  // Check counts
  /*
  for (int m = 0; m < 3; m++) {
    cout << "From process " << proc_id << " Mode " << m;
    cout << " Counts: ";
    for (int i = 0; i < num_procs; i++)
      cout << counts[m][i] / config.rank << " ";
    cout << "\n";

    cout << "From process " << proc_id << " Mode " << m;
    cout << " Disps: ";
    for (int i = 0; i < num_procs; i++)
      cout << disps[m][i] / config.rank << " ";
    cout << "\n";
  }
   */

}


void CoarseTensor::fill_tensor(Config &config) {
  using namespace std;

  FILE* fp = fopen(config.tensor_file.c_str(), "r");
  if (!fp)
    cout << "ERROR: can't open tensor file" << "\n";

  // Allocate memory
  indices = vector<vector<vector<uint64_t>>>(3, vector<vector<uint64_t>>());
  vals = vector<vector<vector<double>>>(3, vector<vector<double>>());

  for (int m = 0; m < 3; m++) {
    indices[m] = vector<vector<uint64_t>>(num_rows[m], vector<uint64_t>());
    vals[m] = vector<vector<double>>(num_rows[m], vector<double>());
  }

  // Parse file
  uint64_t i, j, k;
  double val;
  char* line = NULL;
  ssize_t read;
  size_t len = 0;
  char* ptr = NULL;

  rewind(fp);     // Point to file head
  while ((read = getline(&line, &len, fp)) != -1) {
    ptr = line;
    i = strtoull(ptr, &ptr, 10) - 1;       // 0-based index
    ptr ++;
    j = strtoull(ptr, &ptr, 10) - 1;
    ptr ++;
    k = strtoull(ptr, &ptr, 10) - 1;
    ptr ++;
    val = strtod(ptr, &ptr);

    frob_norm += val * val;

    // Push indices and values if in range
    // WARNING
    if (i < end_rows[0] && i >= start_rows[0]) {
      indices[0][i-start_rows[0]].push_back(k * J + j);
      vals[0][i-start_rows[0]].push_back(val);
    }

    if (j < end_rows[1] && j >= start_rows[1]) {
      indices[1][j-start_rows[1]].push_back(k * I + i);
      vals[1][j-start_rows[1]].push_back(val);
    }

    if (k < end_rows[2] && k >= start_rows[2]) {
      indices[2][k-start_rows[2]].push_back(j * I + i);
      vals[2][k-start_rows[2]].push_back(val);
    }
  }

  frob_norm_sq = sqrt(frob_norm);

  // Close file
  int close = fclose(fp);
  if (close != 0)
    cout << "ERROR: can't close tensor file" << "\n";
}


void CoarseTensor::construct_tensor(Partitioner &partitioner, Config &config) {
  scatter_partition(partitioner, config);
  fill_tensor(config);
}


}   // litetensor
