#ifndef LITETENSOR_TYPES_H
#define LITETENSOR_TYPES_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Cholesky>

namespace litetensor {

// For simplicity, alias for Eigen matrix and vector type
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Mat;
typedef Eigen::VectorXd Vec;

}


#endif // LITETENSOR_TYPES_H
