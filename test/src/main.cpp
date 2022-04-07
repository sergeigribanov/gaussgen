#include <iostream>

#include "gaussgen.hpp"

int main() {
  Eigen::VectorXd x(5);
  x << 1., 2., 3., 4., 5.;
  Eigen::MatrixXd invcov(5, 5);
  invcov << 0.1, 0.01, -0.01, 0.001, 0.001, 0.01, 0.2, 0.03, -0.02, -0.03, -0.01,
      0.03, 0.3, 0.001, 0.00, 0.001, -0.02, 0.001, 0.4, -0.02, 0.001, -0.03,
      0.00, -0.02, 0.5;
  TRandom rnd;
  Eigen::VectorXd result = gaussgen::gaussgen(x, invcov, &rnd);
  std::cout << "cov^-1:" << std::endl;
  std::cout << invcov << std::endl;
  std::cout << "x^T:"  << std::endl;
  std::cout << x.transpose() << std::endl;
  std::cout << "result^T:" << std::endl;
  std::cout << result.transpose() << std::endl;
  return 0;
}
