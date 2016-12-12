# LiteTensor: Parallel and Distributed Sparse Tensor Decomposition 
Student project for CMU 15418/618 Parallel Computer Architecture and Programming.

## Dependencies
* [`OpenMP`](http://www.openmp.org)
* [`OpenMPI`](https://www.open-mpi.org)
* [`Eigen3`](http://eigen.tuxfamily.org/index.php?title=Main_Page) 
Download the Eigen3 zip file from http://bitbucket.org/eigen/eigen/get/3.3.1.zip. I put it in inlcude directory, please modify the Eigen directory in CMakeList. txt if you choose to put it somewhere else.

## Build and Run
Clone this repo, then run the script build.sh to compile the code. 
```
$ git clone https://github.com/Martini09/litetensor
$ cd litetensor; 
$ ./build.sh     
```
The script will put the executable cpd in current directory. To test the code in single machine, run:
```
# Here '-r' means the rank, '-t' means number of threads you want to use.
$ ./cpd -i benchmark/data/tiny.txt -r 2 -t 2   
```
If you want to run MPI code, run the script run_mpi_rocks.sh (which is derived from 15-618 assignment 4, should work in sun grid engine clusters).
```
# '2' means number of nodes, '4' means 4 processors per node
$ ./run_mpi_rocks.sh 2 4 bechmark/data/tiny.txt   
```
