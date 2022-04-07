#include "gaussgen.hpp"

#include <Eigen/Eigenvalues>

Eigen::VectorXd gaussgen::gaussgen(const Eigen::VectorXd& x,
                                   const Eigen::MatrixXd& cov, TRandom* rnd) {
  Eigen::EigenSolver<Eigen::MatrixXd> solver(cov);
  Eigen::VectorXd lambdas = solver.eigenvalues().real();
  Eigen::MatrixXd tm = solver.eigenvectors().real();
  Eigen::MatrixXd tmi = tm.inverse();
  Eigen::VectorXd mean = tmi * x;
  Eigen::VectorXd result = Eigen::VectorXd::Zero(mean.size());
  for (long i = 0; i < mean.size(); ++i) {
    result(i) = rnd->Gaus(mean(i), lambdas(i));
  }
  return tm * result;
}
