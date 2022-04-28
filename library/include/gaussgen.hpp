#include <TRandom.h>

#include <Eigen/Dense>

namespace gaussgen {
  Eigen::VectorXd gaussgen(const Eigen::VectorXd&, const Eigen::MatrixXd&,
                           TRandom*);
  Eigen::VectorXd gaussgen_nfirst(std::size_t, const Eigen::VectorXd&, const Eigen::MatrixXd&,
                           TRandom*);
  Eigen::MatrixXd inverse(const Eigen::MatrixXd&);
}
