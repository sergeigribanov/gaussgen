#include <Eigen/Eigenvalues>
#include "gaussgen.hpp"

Eigen::VectorXd gaussgen::gaussgen(const Eigen::VectorXd& x,
                                   const Eigen::MatrixXd& invcov, TRandom* rnd) {
  Eigen::EigenSolver<Eigen::MatrixXd> solver(invcov);
  Eigen::VectorXd sigma = solver.eigenvalues().real().array().pow(-0.5);
  Eigen::MatrixXd tm = solver.eigenvectors().real();
  Eigen::VectorXd mean = tm.partialPivLu().solve(x);
  Eigen::VectorXd result = Eigen::VectorXd::Zero(mean.size());
  for (long i = 0; i < mean.size(); ++i) {
    result(i) = rnd->Gaus(mean(i), sigma(i));
  }
  return tm * result;
}

Eigen::MatrixXd gaussgen::inverse(const Eigen::MatrixXd& mx) {
  return mx.inverse();
}
