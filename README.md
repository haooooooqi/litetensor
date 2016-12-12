# LiteTensor: Parallel and Distributed Sparse Tensor Decomposition 
Student project for CMU 15418/618 Parallel Computer Architecture and Programming.

## Dependencies
* [`OpenMP`](http://www.openmp.org)
* [`OpenMPI`](https://www.open-mpi.org)
* [`Eigen3`](http://eigen.tuxfamily.org/index.php?title=Main_Page) (Download the zip file from http://eigen.tuxfamily.org/index.php?title=Main_Page. I put it in inlcude directory, please modify the Eigen directory in CMakeList. txt if you choose to put it somewhere else)

## Build and Run
Clone this repo, then run the script build.sh to compile the code. 
```
$ git clone https://github.com/Martini09/litetensor
$ cd litetensor; 
$ ./build.sh     
```
The script will put the executable cpd in current directory.
