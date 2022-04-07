#include <TRandom.h>

#include <Eigen/Dense>

namespace gaussgen {
Eigen::VectorXd gaussgen(const Eigen::VectorXd&, const Eigen::MatrixXd&,
                         TRandom*);
}
